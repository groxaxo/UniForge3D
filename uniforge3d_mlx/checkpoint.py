"""A mixed-precision checkpoint contract with explicit per-matrix formats."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

FORMAT = "uniforge3d-mlx-omlx-oq5-v1"


def read_manifest(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported UniForge3D MLX checkpoint format")
    for name, entry in manifest["tensors"].items():
        shard = Path(entry["shard"])
        if shard.is_absolute() or ".." in shard.parts:
            raise ValueError(f"Unsafe shard path for {name}")
        if "quantization" in entry:
            q = entry["quantization"]
            if q.get("mode") != "affine" or q.get("bits") not in (5, 6, 8) or q.get("group_size") != 64:
                raise ValueError(f"Unsupported quantization for {name}: {q}")
            shape = entry["shape"]
            if len(shape) != 2 or shape[1] % 64:
                raise ValueError(f"Invalid quantized matrix shape for {name}")
    return manifest


def load_checkpoint(root: str | Path, *, names=None) -> dict:
    """Load dense arrays and native packed matrices without dequantizing them."""
    from mlx_spatial.trellis2_quantization import Trellis2QuantizedMatrix

    root = Path(root)
    manifest = read_manifest(root)
    wanted = set(manifest["tensors"]) if names is None else set(names)
    missing = wanted - manifest["tensors"].keys()
    if missing:
        raise KeyError(f"Missing checkpoint tensors: {sorted(missing)[:5]}")
    shards = {}
    for name in wanted:
        entry = manifest["tensors"][name]
        shards.setdefault(entry["shard"], []).append(name)
    result = {}
    for shard, logical_names in shards.items():
        arrays = mx.load(str(root / shard))
        for name in logical_names:
            entry = manifest["tensors"][name]
            if "quantization" in entry:
                q = entry["quantization"]
                qweight, scales, biases = (arrays[name + suffix] for suffix in (".__qweight", ".__scales", ".__biases"))
                rows, cols = entry["shape"]
                if qweight.shape != (rows, cols * q["bits"] // 32) or scales.shape != (rows, cols // 64) or biases.shape != scales.shape:
                    raise ValueError(f"Packed shape mismatch for {name}")
                if qweight.dtype != mx.uint32:
                    raise ValueError(f"Packed weight is not uint32: {name}")
                result[name] = Trellis2QuantizedMatrix(
                    # Stored BF16 parameters are exact when widened to FP32.
                    # Use FP32 arithmetic for geometry-model matrix products.
                    qweight=qweight, scales=scales.astype(mx.float32), biases=biases.astype(mx.float32),
                    original_shape=tuple(entry["shape"]), original_dtype=entry["source_dtype"],
                    bits=q["bits"], group_size=q["group_size"], mode=q["mode"],
                )
            else:
                array = arrays[name]
                if tuple(array.shape) != tuple(entry["shape"]):
                    raise ValueError(f"Dense shape mismatch for {name}")
                result[name] = array
    return result


def linear(x, weights: dict, prefix: str):
    """Use the same packed matrix operation as the existing MLX TRELLIS port."""
    from mlx_spatial.trellis2_quantization import trellis2_linear
    return trellis2_linear(x, weights[prefix + ".weight"], weights.get(prefix + ".bias"))
