#!/usr/bin/env python3
"""Real two-view conditioning smoke test and independent PyTorch point-head parity."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np

from uniforge3d_mlx.checkpoint import load_checkpoint, read_manifest
from uniforge3d_mlx.dinov3 import DinoV3Encoder
from uniforge3d_mlx.flow import RoutedFlow
from uniforge3d_mlx.ops import release_memory
from uniforge3d_mlx.preprocessing import preprocess
from uniforge3d_mlx.routing import route_views
from uniforge3d_mlx.vggt import VGGT, adapt_features


def comparison(reference, candidate):
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(candidate, dtype=np.float64).ravel()
    return {
        "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
        "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "max_absolute_error": float(np.max(np.abs(a - b))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--bundle", type=Path, default=Path("weights/UniForge3D-oQ5-complete")
    )
    args = parser.parse_args()
    if len(args.images) < 2:
        parser.error("Provide at least two distinct input views")
    mx.set_cache_limit(1024**3)
    config = json.loads(Path("uniforge3d/configs/pipeline.json").read_text())
    out = args.run / "multiview-validation"
    out.mkdir(exist_ok=True)
    batch = preprocess(
        args.images,
        config["image_preprocess"],
        args.bundle,
        preview_dir=out,
        log=lambda s: print(s, flush=True),
    )
    with np.load(args.run / "structure.npz") as data:
        c32, c64 = data["coords32"], data["coords64"]
    print("Running MLX DINOv3 for multiple views", flush=True)
    dino_model = DinoV3Encoder(args.bundle / "dinov3")
    dino = dino_model(batch.dino512)
    mx.eval(dino)
    del dino_model
    release_memory()
    print("Running shape-conditioned VGGT and DPT point head", flush=True)
    model = VGGT(args.bundle / "vggt_conditioner", conditioner=True)
    cached, points, confidence = model.conditioned_features(
        batch.vggt, mx.array(c64[:, 1:]), return_points=True
    )
    mx.eval(points, confidence)
    shape_tokens = model.shape_tokens(mx.array(c64[:, 1:]))
    mx.eval(shape_tokens)
    np.savez(
        out / "point-head-inputs.npz",
        shape_tokens=np.array(shape_tokens),
        images=np.array(batch.vggt),
        **{f"layer_{key}": np.array(value) for key, value in cached.items()},
    )
    point_mlx, confidence_mlx = np.array(points), np.array(confidence)
    selected = route_views(point_mlx, confidence_mlx, batch.masks, c32, grid=32)
    adapter = load_checkpoint(args.bundle / "shape512_adapter")
    delta = adapt_features(cached, adapter)
    bank = mx.concatenate((dino[:, :, :5], dino[:, :, 5:] + delta), 2)
    mx.eval(bank)
    del model, adapter, cached, shape_tokens, points, confidence, delta, dino
    release_memory()
    print(
        "Running one routed Shape512 prediction with the actual view assignments",
        flush=True,
    )
    model = RoutedFlow(
        args.bundle / "shape512",
        config["structured_latent_512"]["model"]["args"]["denoiser_args"],
        c32,
        bank,
        selected,
    )
    noise = mx.array(
        np.random.default_rng(42).standard_normal((len(c32), 32), dtype=np.float32)
    )
    prediction = model(noise, 1.0, True)
    mx.eval(prediction)
    finite = bool(mx.all(mx.isfinite(prediction)).item())
    np.savez(
        out / "mlx-output.npz",
        points=point_mlx,
        confidence=confidence_mlx,
        selected=selected,
        prediction=np.array(prediction),
    )
    del model, noise, prediction, bank
    release_memory()

    print("Checking the full point head against original PyTorch on CPU", flush=True)
    import torch

    for package, relative in (
        ("uniforge3d", "uniforge3d"),
        ("uniforge3d.models", "uniforge3d/models"),
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(Path(relative).resolve())]
        sys.modules[package] = module
    source = importlib.import_module("uniforge3d.models.vggt_routed_features")
    ref = types.SimpleNamespace(
        point_head=source.DPTHead(
            dim_in=2048, output_dim=4, activation="inv_log", conf_activation="expp1"
        ).eval(),
        dpt_shape_adapter_32=source.DPTShapeAdapter32().eval(),
    )
    names = [
        k
        for k in read_manifest(args.bundle / "vggt_conditioner")["tensors"]
        if k.startswith(("point_head.", "dpt_shape_adapter_32."))
    ]
    weights = load_checkpoint(args.bundle / "vggt_conditioner", names=names)
    for prefix, module in (
        ("point_head.", ref.point_head),
        ("dpt_shape_adapter_32.", ref.dpt_shape_adapter_32),
    ):
        module.load_state_dict(
            {
                key.removeprefix(prefix): torch.from_numpy(
                    np.array(value.astype(mx.float32))
                )
                for key, value in weights.items()
                if key.startswith(prefix)
            },
            strict=True,
        )
    del weights
    release_memory()
    with np.load(out / "point-head-inputs.npz") as data:
        tensors = {key: torch.from_numpy(data[key]) for key in data.files}
    cached = [tensors.get(f"layer_{i}") for i in range(24)]
    with torch.inference_mode():
        point_ref, confidence_ref = (
            source.FrozenRoutedVGGTFeatureExtractor._point_forward(
                ref,
                cached,
                tensors["images"],
                5,
                # MLX uses an explicit batch axis; the original packed API
                # takes [total_shape_tokens, channels] plus batch offsets.
                tensors["shape_tokens"][0],
                torch.tensor([0, len(c64)], dtype=torch.long),
            )
        )
    rows = {
        "points": comparison(point_ref.numpy(), point_mlx),
        "confidence": comparison(confidence_ref.numpy(), confidence_mlx),
    }
    report = {
        "views": len(args.images),
        "target_voxels": len(c32),
        "finite_prediction": finite,
        "routed_counts": np.bincount(selected, minlength=len(args.images)).tolist(),
        "point_head_parity": rows,
        "passed": finite
        and all(
            row["cosine"] >= 0.99999 and row["relative_l2"] <= 0.003
            for row in rows.values()
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
