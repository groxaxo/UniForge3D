#!/usr/bin/env python3
"""Run UniForge3D locally on Apple Silicon, without importing the CUDA runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--bundle", type=Path, default=Path("weights/UniForge3D-oQ5-complete")
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="A new run directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--high-res-precision",
        choices=("float32", "mixed"),
        default="mixed",
        help="High-resolution flow arithmetic: validated mixed mode (default) or FP32",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override all stages for diagnostics; defaults use the trained schedule",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching, hash-verified stage artifacts",
    )
    parser.add_argument(
        "--stop-after",
        choices=("structure", "shape512", "handoff", "shape1024", "texture", "decode"),
    )
    args = parser.parse_args()
    import mlx.core as mx
    from uniforge3d_mlx.pipeline import Pipeline

    mx.set_cache_limit(1024**3)
    mx.set_memory_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    pipeline = Pipeline(
        args.bundle,
        args.output,
        config_path=args.config,
        seed=args.seed,
        steps=args.steps,
        resume=args.resume,
        high_res_precision=args.high_res_precision,
        log=lambda message: print(message, flush=True),
    )
    result = pipeline.run(args.images, stop_after=args.stop_after)
    print(
        f"GLB: {result}" if result else f"Stage checkpoint: {args.output}", flush=True
    )


if __name__ == "__main__":
    main()
