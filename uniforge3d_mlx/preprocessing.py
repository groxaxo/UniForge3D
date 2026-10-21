"""Shared alpha crop and image tensors, matching the source pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .checkpoint import load_checkpoint
from .ops import release_memory


@dataclass
class InputBatch:
    dino512: mx.array
    dino1024: mx.array
    vggt: mx.array
    masks: np.ndarray


def crop_with_padding(array, left, top, side):
    output = np.zeros((side, side, *array.shape[2:]), dtype=array.dtype)
    height, width = array.shape[:2]
    l, t, r, b = (
        max(left, 0),
        max(top, 0),
        min(left + side, width),
        min(top + side, height),
    )
    if r > l and b > t:
        output[t - top : b - top, l - left : r - left] = array[t:b, l:r]
    return output


def preprocess(images, config, bundle, *, preview_dir=None, log=print):
    from mlx_spatial.trellis2_rmbg_forward import _BiRefNetMlx, _prepare_rmbg2_input

    tensors = None
    dino512, dino1024, vggt, masks = [], [], [], []
    for index, path in enumerate(images):
        with Image.open(path) as source:
            image = source.convert("RGBA")
            if source.mode != "RGBA" or np.all(
                np.asarray(image.getchannel("A")) == 255
            ):
                log(f"Removing background from view {index + 1}")
                if tensors is None:
                    tensors = load_checkpoint(Path(bundle) / "rmbg")
                rgb = image.convert("RGB")
                logits = _BiRefNetMlx(tensors)(_prepare_rmbg2_input(rgb))
                mask = mx.sigmoid(logits[0, 0])
                mx.eval(mask)
                alpha = Image.fromarray(
                    (np.clip(np.array(mask), 0, 1) * 255).astype(np.uint8)
                ).resize(rgb.size)
                image.putalpha(alpha)
                del logits, mask
            rgba = np.asarray(image, dtype=np.float32) / 255
        alpha = rgba[..., 3]
        ys, xs = np.nonzero(alpha > config["alpha_threshold"])
        if len(xs) == 0:
            raise ValueError(f"{path}: empty alpha mask")
        side = max(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        side = max(1, math.ceil(side * config["padding_factor"]))
        left = math.floor((int(xs.min()) + int(xs.max()) + 1 - side) * 0.5)
        top = math.floor((int(ys.min()) + int(ys.max()) + 1 - side) * 0.5)
        rgb = (rgba[..., :3] * alpha[..., None] * 255).clip(0, 255).astype(np.uint8)
        crop = Image.fromarray(crop_with_padding(rgb, left, top, side))
        for size, items in ((512, dino512), (1024, dino1024), (448, vggt)):
            pixels = (
                np.asarray(
                    crop.resize((size, size), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
                / 255
            )
            items.append(pixels.transpose(2, 0, 1))
        a = crop_with_padding(alpha, left, top, side)
        a = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).resize(
            (448, 448), Image.Resampling.NEAREST
        )
        masks.append(np.asarray(a) > 127)
        if preview_dir is not None:
            destination = Path(preview_dir)
            destination.mkdir(parents=True, exist_ok=True)
            image.save(destination / f"view-{index + 1:02d}-rgba.png")
            crop.save(destination / f"view-{index + 1:02d}-crop.png")
    del tensors
    release_memory()
    return InputBatch(
        mx.array(np.stack(dino512)[None]),
        mx.array(np.stack(dino1024)[None]),
        mx.array(np.stack(vggt)[None]),
        np.stack(masks)[None],
    )
