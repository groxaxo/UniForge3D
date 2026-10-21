#!/usr/bin/env python3
"""Exercise the bundled background remover and compare its mask with an RGBA input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uniforge3d_mlx.checkpoint import load_checkpoint
from uniforge3d_mlx.ops import release_memory
import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_spatial.trellis2_rmbg_forward import _BiRefNetMlx, _prepare_rmbg2_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--bundle", type=Path, default=Path("weights/UniForge3D-oQ5-complete")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGBA")
    original = np.asarray(image.getchannel("A")) > 127
    weights = load_checkpoint(args.bundle / "rmbg")
    start = time.perf_counter()
    logits = _BiRefNetMlx(weights)(_prepare_rmbg2_input(image.convert("RGB")))
    mask = mx.sigmoid(logits[0, 0])
    mx.eval(mask)
    elapsed = time.perf_counter() - start
    mask_values = np.array(mask)
    finite = bool(np.isfinite(mask_values).all())
    alpha = Image.fromarray((np.clip(mask_values, 0, 1) * 255).astype(np.uint8)).resize(
        image.size
    )
    predicted = np.asarray(alpha) > 127
    iou = float((original & predicted).sum() / max(1, (original | predicted).sum()))
    image.putalpha(alpha)
    image.save(args.output / "rmbg-result.png")
    fraction = float(predicted.mean())
    report = {
        "finite": finite,
        "seconds": elapsed,
        "foreground_fraction": fraction,
        "iou_with_supplied_alpha": iou,
        "passed": finite and 0.01 < fraction < 0.99 and iou >= 0.8,
        "note": "Supplied alpha is an external segmentation reference, not a paired FP32 model output.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    del weights, logits, mask
    release_memory()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
