"""Selective oMLX oQ5 allocation for UniForge3D's supported matrix operations.

The scores below measure weight reconstruction, not image/geometry quality or
activation-calibrated sensitivity. New adapters and prediction boundaries are
excluded rather than being treated as interchangeable transformer weights.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

import mlx.core as mx
import numpy as np

from .checkpoint import FORMAT
from .source import tensor_to_mlx

OMLX_COMMIT = "70f44cfd652d2013dc0f9776dca7518e32101b00"
_DIT = re.compile(r"^blocks\.(\d+)\.(?:self_attn\.(?:to_qkv|to_out)|cross_attn\.(?:to_q|to_kv|to_out)|mlp\.mlp\.(?:0|2))\.weight$")
_VGGT = re.compile(r"^aggregator\.(?:(?:frame|global)_blocks\.(\d+)|patch_embed\.blocks\.(\d+))\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))\.weight$")
_DINO = re.compile(r"^layer\.(\d+)\.(?:attention\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:up_proj|down_proj))\.weight$")
_RMBG = re.compile(r"^bb\.layers\.(\d+)\.blocks\.(\d+)\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2))\.weight$")


def eligible(component: str, name: str, shape: tuple, dtype: str) -> bool:
    if len(shape) != 2 or shape[1] % 64 or dtype not in ("float16", "bfloat16", "float32"):
        return False
    if component in ("structure_dit", "shape512", "shape1024", "texture"):
        return _DIT.fullmatch(name) is not None
    if component in ("structure_vggt", "vggt_conditioner"):
        return _VGGT.fullmatch(name) is not None
    if component == "dinov3":
        return _DINO.fullmatch(name) is not None
    if component == "rmbg":
        return _RMBG.fullmatch(name) is not None
    return False


def import_omlx(root: Path):
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != OMLX_COMMIT:
        raise RuntimeError(f"oMLX converter changed: {commit}; expected {OMLX_COMMIT}")
    sys.path.insert(0, str(root))
    oq = importlib.import_module("omlx.oq")
    if not oq.HAS_MLX or 5 not in oq.OQ_LEVELS:
        raise RuntimeError("The selected oMLX build cannot perform MLX oQ5 quantization")
    return oq


def array_sha256(array) -> str:
    mx.eval(array)
    # BF16 cannot be represented by NumPy directly; hash the stored bytes.
    raw = np.asarray(array.reshape(-1).view(mx.uint8))
    return hashlib.sha256(raw.tobytes()).hexdigest()


def reconstruction_error(weight, oq, bits=5, max_rows=64, importance=None):
    rows = min(int(weight.shape[0]), max_rows)
    indices = mx.array(np.linspace(0, weight.shape[0] - 1, rows, dtype=np.int32))
    sample = weight[indices].astype(mx.float32)
    if importance is not None:
        importance = mx.array(importance, dtype=mx.float32)
        if importance.shape != (weight.shape[1],) or not bool(mx.all(mx.isfinite(importance)).item()) or not bool(mx.all(importance >= 0).item()):
            raise ValueError("Invalid activation importance vector")
    qw, scales, biases = oq._quantize_chunked(sample.astype(mx.bfloat16), 64, bits, "affine", importance=importance)
    restored = mx.dequantize(qw, scales.astype(mx.float32), biases.astype(mx.float32), group_size=64, bits=bits)
    weighted = importance if importance is not None else 1.0
    mse = mx.mean(weighted * mx.square(restored - sample))
    power = mx.mean(weighted * mx.square(sample))
    relative = mse / mx.maximum(power, mx.array(1e-20))
    mx.eval(relative)
    value = float(relative.item())
    if not np.isfinite(value):
        raise ValueError("Non-finite quantization reconstruction error")
    return value


def make_plan(source, component: str, oq):
    names = [n for n in source if eligible(component, n, source.shape(n), source.dtype(n))]
    allowlist = getattr(source, "quantization_allowlist", None)
    if allowlist is not None:
        invalid = set(allowlist) - set(names)
        if invalid:
            raise ValueError(f"Quantization allow-list contains unsupported matrices: {sorted(invalid)[:3]}")
        names = [n for n in names if n in allowlist]
    if not names:
        return {}, {}, None
    shapes, scores, virtual_to_real = {}, {}, {}
    for i, name in enumerate(names):
        weight = tensor_to_mlx(source.tensor(name))
        importance = getattr(source, "importance", {}).get(name)
        error = reconstruction_error(weight, oq, importance=importance)
        # oMLX's generic allocator indexes priority scores through layers.<n>.
        # One allocator slot per matrix enables tensor-level ranking here; it
        # does not describe or impersonate a language-model architecture.
        virtual = f"model.layers.{i}.matrix"
        shapes[virtual] = source.shape(name)
        scores[str(i)] = max(error, 1e-12)
        virtual_to_real[virtual] = name
        del weight
        mx.clear_cache()
        if i % 40 == 0:
            print(f"SENSITIVITY {component} {i + 1}/{len(names)}", flush=True)
    config = {"num_hidden_layers": len(names), "_oq_sensitivity_map": scores,
              "_oq_use_budget_plan": True}
    plan = oq._build_quant_plan(shapes, config, oq_level=5, target_bpw=5.5,
                               hard_cap_bpw=5.7, supported_bits=(5, 6, 8))
    result = {virtual_to_real[p]: plan.boost_map.get(p, {"bits": 5, "group_size": 64, "mode": "affine"})
              for p in shapes}
    errors = {virtual_to_real[p]: scores[str(i)] for i, p in enumerate(shapes)}
    return result, errors, {"effective_bpw_selected_matrices": plan.effective_bpw,
                           "target_bpw": plan.target_bpw, "hard_cap_bpw": plan.hard_cap_bpw}


def convert_component(source, component: str, output_root: Path, *, omlx_root: Path,
                      source_paths=(), max_shard_bytes=256 * 1024 * 1024):
    oq = import_omlx(omlx_root)
    destination = output_root / component
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing artifact: {destination}")
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    plan, errors, allocation = make_plan(source, component, oq)
    manifest = {
        "format": FORMAT, "component": component, "oq_level": 5,
        "omlx_root": str(omlx_root), "omlx_commit": OMLX_COMMIT,
        "policy": getattr(source, "quantization_policy", "internal-transformer-only-source-precision-boundaries-v1"),
        "sensitivity": "deterministic 64-row weight-reconstruction relative MSE; not activation calibration",
        "allocation": allocation, "sources": [str(p) for p in source_paths],
        "merged_lora_pairs": len(getattr(source, "lora", {})),
        "tensors": {},
    }
    if getattr(source, "importance", None):
        if set(source.importance) != set(plan):
            raise ValueError("Activation statistics must cover every eligible matrix exactly")
        manifest["sensitivity"] = "activation-diagonal-weighted relative MSE on 64 deterministic rows"
        manifest["calibration"] = source.calibration
    shard_data, shard_names = {}, []
    shard_bytes, shard_index, source_bytes, payload_bytes = 0, 0, 0, 0
    bits_count, params_count = Counter(), Counter()

    def flush():
        nonlocal shard_data, shard_names, shard_bytes, shard_index
        if not shard_data:
            return
        filename = f"weights-{shard_index:04d}.safetensors"
        path = staging / filename
        mx.save_safetensors(str(path), shard_data, metadata={"format": FORMAT, "component": component})
        for name in shard_names:
            manifest["tensors"][name]["shard"] = filename
        shard_index += 1
        shard_data, shard_names, shard_bytes = {}, [], 0
        mx.synchronize()
        mx.clear_cache()

    for i, name in enumerate(source):
        value = tensor_to_mlx(source.tensor(name))
        mx.eval(value)
        source_bytes += value.nbytes
        entry = {"shape": list(value.shape), "source_dtype": source.dtype(name)}
        if name in plan:
            q = plan[name]
            importance = getattr(source, "importance", {}).get(name)
            qw, scales, biases = oq._quantize_chunked(value.astype(mx.bfloat16), 64, q["bits"], "affine", importance=importance)
            mx.eval(qw, scales, biases)
            if not bool(mx.all(mx.isfinite(scales)).item()) or not bool(mx.all(mx.isfinite(biases)).item()):
                raise ValueError(f"Non-finite packed scales for {name}")
            entry["quantization"] = dict(q)
            entry["sample_relative_mse_q5"] = errors[name]
            entry["sample_relative_mse_output"] = reconstruction_error(value, oq, bits=q["bits"], importance=importance)
            for suffix, array in ((".__qweight", qw), (".__scales", scales), (".__biases", biases)):
                shard_data[name + suffix] = array
                shard_bytes += array.nbytes
                payload_bytes += array.nbytes
            bits_count[str(q["bits"])] += 1
            params_count[str(q["bits"])] += value.size
        else:
            # A byte-preserving passthrough, including FP32 DPT heads/adapters.
            shard_data[name] = value
            shard_bytes += value.nbytes
            payload_bytes += value.nbytes
            entry["passthrough_sha256"] = array_sha256(value)
            bits_count["source_precision"] += 1
            params_count["source_precision"] += value.size
        manifest["tensors"][name] = entry
        shard_names.append(name)
        del value
        if shard_bytes >= max_shard_bytes:
            flush()
        if i % 100 == 0:
            print(f"WRITE {component} {i + 1} tensors", flush=True)
    flush()
    manifest.update({"tensor_count": len(manifest["tensors"]), "matrix_counts": dict(bits_count),
                     "parameter_counts": dict(params_count), "source_logical_bytes": source_bytes,
                     "output_payload_bytes": payload_bytes, "seconds": time.time() - started})
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(staging, destination)
    print(f"CONVERTED {component}: {source_bytes / 1e9:.3f} -> {payload_bytes / 1e9:.3f} GB {dict(bits_count)}", flush=True)
    return manifest
