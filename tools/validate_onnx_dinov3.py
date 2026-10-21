#!/usr/bin/env python3
"""Validate recovered DINOv3 against actual ONNX inference on frozen images."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uniforge3d_mlx.onnx_dinov3 import sha256
from tools.validate_softeq_dinov3 import metrics

SOURCE = ROOT / "weights/source/dinov3-onnx-hf"
WORK = ROOT / "weights/work/onnx-validation"
CASES = (("case01", "calibration"), ("case03", "calibration"),
         ("case02", "held_out"), ("case04", "held_out"))


def write(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n")


def frozen_cases(sizes):
    from PIL import Image
    import numpy as np
    for name, split in CASES:
        path = ROOT / "assets" / name / "001.png"
        with Image.open(path) as image:
            image = image.convert("RGB")
            for size in sizes:
                pixels = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS),
                                    dtype=np.float32).transpose(2, 0, 1)[None] / 255
                yield {"case": name, "split": split, "size": size,
                       "image_sha256": sha256(path)}, pixels


def normalize(pixels):
    import numpy as np
    mean = np.array([.485, .456, .406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([.229, .224, .225], dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((pixels - mean) / std)


def source_gate(test):
    return test["cosine"] >= .9999 and test["minimum_token_cosine"] >= .999


def candidate_gate(test):
    return test["cosine"] >= .999 and test["minimum_token_cosine"] >= .995


def reference_path(info):
    return WORK / f"reference-{info['case']}-{info['size']}.npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("onnx", "source", "candidate"), required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024])
    args = parser.parse_args()
    import numpy as np
    WORK.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((SOURCE / "preparation.json").read_text())
    if sha256(SOURCE / "model.safetensors") != preparation["normalized_sha256"]:
        raise ValueError("Recovered source changed")
    cases = list(frozen_cases(args.sizes))
    identity = [info for info, _ in cases]
    reference_report_path = WORK / "onnx-reference.json"
    result = {"source_sha256": preparation["normalized_sha256"], "inputs": identity,
              "tests": [], "fidelity_gate": {"cosine": .999, "minimum_token_cosine": .995},
              "scope": "DINO encoder features; four images, two calibration and two held out; no 3D generation"}
    if args.phase == "onnx":
        import onnxruntime as ort
        graph = Path(preparation["reference_graph"])
        if sha256(graph) != preparation["reference_graph_sha256"]:
            raise ValueError("ONNX reference graph changed")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(graph), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        result.update({"onnxruntime": ort.__version__, "providers": session.get_providers(),
                       "graph_sha256": sha256(graph)})
        for info, pixels in cases:
            print("ONNX", info, flush=True)
            started = time.perf_counter()
            public_output, raw = session.run(
                ["last_hidden_state", preparation["reference_raw_output"]],
                {"pixel_values": normalize(pixels)})
            wanted = (1, 5 + (info["size"] // 16) ** 2, 1024)
            if public_output.shape != wanted or raw.shape != wanted or not np.isfinite(raw).all() or not np.isfinite(public_output).all():
                raise ValueError("ONNX inference returned invalid features")
            # UniForge3D deliberately replaces the trained final affine norm.
            raw64 = raw.astype(np.float64)
            features = ((raw64 - raw64.mean(-1, keepdims=True)) /
                        np.sqrt(raw64.var(-1, keepdims=True) + 1e-5)).astype(np.float32)
            path = reference_path(info)
            np.save(path, features)
            test = {**info, "shape": list(features.shape), "seconds": time.perf_counter() - started,
                    "reference_sha256": sha256(path)}
            result["tests"].append(test)
            write(WORK / "onnx-progress.json", result)
            print(json.dumps(test), flush=True)
            del raw64, raw, public_output, features
            gc.collect()
        result["all_finite"] = True
        write(reference_report_path, result)
        return

    reference = json.loads(reference_report_path.read_text())
    if reference["inputs"] != identity or reference["source_sha256"] != result["source_sha256"]:
        raise ValueError("Frozen reference identity mismatch")
    for test in reference["tests"]:
        if sha256(reference_path(test)) != test["reference_sha256"]:
            raise ValueError("Frozen ONNX reference changed")
    if args.phase == "source":
        import torch
        from safetensors.torch import load_file
        from uniforge3d_mlx.softeq import torch_backbone, torch_features
        from uniforge3d_mlx.quantization import eligible
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required for this source comparison")
        torch.mps.empty_cache()
        config = json.loads((SOURCE / "config.json").read_text())
        model = torch_backbone(config, load_file(str(SOURCE / "model.safetensors"))).to("mps")
        collected, handles = {}, []
        calibrating = False

        def hook_for(name):
            def capture(module, values):
                if not calibrating:
                    return
                values = values[0].detach().float().reshape(-1, values[0].shape[-1])
                total, count = collected.get(name, (0, 0))
                collected[name] = (total + values.square().sum(0), count + values.shape[0])
            return capture

        for name, module in model.named_modules():
            key = name.removeprefix("model.") + ".weight"
            if isinstance(module, torch.nn.Linear) and eligible("dinov3", key, tuple(module.weight.shape), "float32"):
                handles.append(module.register_forward_pre_hook(hook_for(key)))
        with torch.inference_mode():
            for info, pixels in cases:
                calibrating = info["split"] == "calibration"
                print("TORCH MPS", info, flush=True)
                started = time.perf_counter()
                output = torch_features(model, torch.from_numpy(normalize(pixels)).to("mps")).cpu().numpy()
                test = {**info, **metrics(np.load(reference_path(info)), output),
                        "seconds": time.perf_counter() - started}
                result["tests"].append(test)
                print(json.dumps(test), flush=True)
                if not source_gate(test):
                    write(WORK / "torch-parity-failed.json", result)
                    raise RuntimeError("Recovered PyTorch model differs from ONNX")
        for handle in handles:
            handle.remove()
        importance = {k: (total / count).cpu().numpy() for k, (total, count) in collected.items()}
        if len(importance) != 144:
            raise ValueError("Calibration must cover all 144 internal matrices")
        np.savez(WORK / "importance.npz", **importance)
        calibration = {
            "method": "FP32 input activation mean-square diagonal",
            "inputs": [info for info in identity if info["split"] == "calibration"],
            "held_out_used_for_calibration": False, "matrix_count": len(importance),
            "token_counts": {k: count for k, (_, count) in collected.items()},
            "source_sha256": result["source_sha256"],
            "importance_sha256": sha256(WORK / "importance.npz"),
        }
        write(WORK / "calibration.json", calibration)
        result["all_passed"] = True
        result["torch"] = torch.__version__
        write(WORK / "torch-parity.json", result)
        del model, module, collected, importance, output, handles
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        print("MPS memory released", torch.mps.current_allocated_memory(),
              torch.mps.driver_allocated_memory(), flush=True)
        result["tests"] = []

    import mlx.core as mx
    from uniforge3d_mlx.dinov3 import DinoV3Encoder
    mx.clear_cache()
    mx.set_cache_limit(128 * 1024 * 1024)
    if args.phase == "candidate" and (args.candidate is None or not args.label):
        raise ValueError("Candidate path and label are required")
    root = SOURCE if args.phase == "source" else args.candidate
    encoder = DinoV3Encoder(root)
    result.update({"phase": args.phase, "candidate": str(root),
                   "mlx": importlib.metadata.version("mlx")})
    held_out = []
    for info, pixels in cases:
        print("MLX", args.phase, args.label or "fp32", info, flush=True)
        started = time.perf_counter()
        output = encoder(pixels)
        mx.eval(output)
        actual = np.asarray(output)
        test = {**info, **metrics(np.load(reference_path(info)), actual),
                "seconds": time.perf_counter() - started}
        test["passed"] = source_gate(test) if args.phase == "source" else candidate_gate(test)
        result["tests"].append(test)
        print(json.dumps(test), flush=True)
        if info["split"] == "held_out" and info["size"] == 512:
            held_out.append((pixels, actual.copy()))
        if args.phase == "source" and not test["passed"]:
            write(WORK / "source-parity-failed.json", result)
            raise RuntimeError("MLX source encoder differs from ONNX")
    result["all_passed"] = all(t["passed"] for t in result["tests"])
    if result["all_passed"] and args.phase == "candidate" and len(held_out) == 2:
        multiview_input = np.stack([p[0] for p, _ in held_out])[None]
        output = encoder(multiview_input)
        mx.eval(output)
        expected = np.stack([a[0] for _, a in held_out])[None]
        result["multiview"] = metrics(expected, np.asarray(output))
        if not source_gate(result["multiview"]):
            raise RuntimeError("Multiview batch differs from individual views")
    result["peak_mlx_memory_bytes"] = mx.get_peak_memory()
    label = "source" if args.phase == "source" else args.label
    write(WORK / f"{label}-parity.json", result)
    del encoder, output, actual
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    print("MLX memory released", mx.get_active_memory(), mx.get_cache_memory(), flush=True)


if __name__ == "__main__":
    main()
