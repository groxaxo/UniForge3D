#!/usr/bin/env python3
"""Compare complete 30-block high-resolution flow predictions at matched inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np

from uniforge3d_mlx.checkpoint import load_checkpoint
from uniforge3d_mlx.flow import RoutedFlow
from uniforge3d_mlx.ops import release_memory
from uniforge3d_mlx.vggt import adapt_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, default=Path("weights/UniForge3D-oQ5-complete")
    )
    parser.add_argument(
        "--component", choices=("shape1024", "texture"), default="shape1024"
    )
    args = parser.parse_args()
    mx.set_cache_limit(1024**3)
    config = json.loads(Path("uniforge3d/configs/pipeline.json").read_text())
    with np.load(args.run / "handoff.npz") as data:
        coords = data["coords64"]
    with np.load(args.run / "dino.npz") as data:
        dino = mx.array(data["features1024"])
    with np.load(args.run / "vggt_shape1024.npz") as data:
        cached = {i: mx.array(data[f"layer_{i}"]) for i in (4, 11, 17, 23)}
    w = load_checkpoint(args.bundle / (args.component + "_adapter"))
    delta = adapt_features(cached, w, upsample=True)
    bank = mx.concatenate((dino[:, :, :5], dino[:, :, 5:] + delta), 2)
    mx.eval(bank)
    del w, cached, delta, dino
    release_memory()
    rng = np.random.default_rng(
        np.random.SeedSequence([42, 2 if args.component == "shape1024" else 3])
    )
    noise = mx.array(rng.standard_normal((len(coords), 32), dtype=np.float32))
    conditions = [(1.0, True), (0.35, True), (0.35, False)]
    outputs = {}
    timings = {}
    concat = None
    config_key = (
        "structured_latent_1024" if args.component == "shape1024" else "texture"
    )
    if args.component == "texture":
        with np.load(args.run / "shape1024.npz") as data:
            normalization = config["texture"]["shape_slat_normalization"]
            concat = (
                mx.array(data["vae_features"]) - mx.array(normalization["mean"])
            ) / mx.array(normalization["std"])
    for precision in ("float32", "mixed"):
        print(f"Testing {precision} on {len(coords):,} tokens", flush=True)
        model = RoutedFlow(
            args.bundle / args.component,
            config[config_key]["model"]["args"]["denoiser_args"],
            coords,
            bank,
            np.zeros(len(coords), dtype=np.int32),
            compute_dtype=precision,
            concat_features=concat,
        )
        outputs[precision] = []
        timings[precision] = []
        for t, conditioned in conditions:
            start = time.perf_counter()
            value = model(noise, t, conditioned)
            mx.eval(value)
            seconds = time.perf_counter() - start
            outputs[precision].append(np.array(value))
            timings[precision].append(seconds)
            print(
                f"{precision}: t={t}, conditioned={conditioned}, {seconds:.2f}s",
                flush=True,
            )
        del model, value
        release_memory()
    rows = []
    for i, (t, conditioned) in enumerate(conditions):
        a = outputs["float32"][i].astype(np.float64).ravel()
        b = outputs["mixed"][i].astype(np.float64).ravel()
        row = {
            "t": t,
            "conditioned": conditioned,
            "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            "max_absolute_error": float(np.max(np.abs(a - b))),
            "reference_seconds": timings["float32"][i],
            "mixed_seconds": timings["mixed"][i],
        }
        row["passed"] = bool(
            np.isfinite(b).all()
            and row["cosine"] >= 0.99999
            and row["relative_l2"] <= 0.005
        )
        rows.append(row)
    report = {
        "component": args.component,
        "tokens": len(coords),
        "rows": rows,
        "gate": {"cosine_min": 0.99999, "relative_l2_max": 0.005},
        "passed": all(row["passed"] for row in rows),
        "median_speedup": float(
            np.median(timings["float32"]) / np.median(timings["mixed"])
        ),
    }
    filename = (
        "flow-precision.json"
        if args.component == "shape1024"
        else "texture-flow-precision.json"
    )
    (args.run / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
