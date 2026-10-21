"""Dense occupancy and sparse FlexiDualGrid/PBR decoding with the pinned MLX backend."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_spatial.trellis2_sparse_structure import (
    SparseStructureDecoderConfig,
    _decode_sparse_structure_latent,
)
from mlx_spatial.trellis2_decode import (
    _linear_chunked,
    _sparse_convnext_block_forward,
    _sparse_c2s_upblock_forward,
)

from .checkpoint import load_checkpoint
from .ops import norm


def decode_structure(bundle, latent):
    config = SparseStructureDecoderConfig(
        "SparseStructureDecoder", 1, 8, 2, (512, 128, 32), 2, "layer", True
    )
    w = load_checkpoint(Path(bundle) / "structure_decoder")
    logits = _decode_sparse_structure_latent(latent, config, w)
    mx.eval(logits)
    occupied = np.array(logits)[0, 0] > 0
    if occupied.shape != (64, 64, 64):
        raise ValueError(f"Expected occupancy at 64^3, got {occupied.shape}")
    coords64 = np.argwhere(occupied).astype(np.int32)
    occupied32 = occupied.reshape(32, 2, 32, 2, 32, 2).max(axis=(1, 3, 5))
    coords32 = np.argwhere(occupied32).astype(np.int32)
    if not len(coords32) or not len(coords64):
        raise ValueError("Sparse structure decoder produced an empty surface")
    return (
        np.pad(coords32, ((0, 0), (1, 0))),
        np.pad(coords64, ((0, 0), (1, 0))),
        np.array(logits),
    )


def decode_sparse(
    bundle,
    coords,
    features,
    *,
    texture=False,
    guide=None,
    max_tokens=12_000_000,
    log=print,
):
    """Evaluate and free each block; preserve the shape decoder's row ordering."""
    name = "texture_decoder" if texture else "shape_decoder"
    w = load_checkpoint(Path(bundle) / name)
    channels = (1024, 512, 256, 128, 64)
    blocks = (4, 16, 8, 4, 0)
    coords = mx.array(coords, dtype=mx.int32)
    x = _linear_chunked(
        features.astype(mx.float32), w["from_latent.weight"], w["from_latent.bias"]
    )
    subdivisions = []
    for level, count in enumerate(blocks):
        if len(coords) > max_tokens:
            raise MemoryError(
                f"{name} level {level}: {len(coords)} voxels exceeds budget {max_tokens}"
            )
        log(
            f"{name} level {level + 1}/5: {len(coords):,} voxels, {channels[level]} channels"
        )
        for index in range(count):
            x = _sparse_convnext_block_forward(
                coords, x, w, prefix=f"blocks.{level}.{index}"
            )
            mx.eval(x)
        if level < 4:
            coords, x, sub = _sparse_c2s_upblock_forward(
                coords,
                x,
                w,
                prefix=f"blocks.{level}.{count}",
                in_channels=channels[level],
                out_channels=channels[level + 1],
                pred_subdiv=not texture,
                guide_subdivision=guide[level] if texture else None,
            )
            mx.eval(coords, x, sub)
            subdivisions.append(sub)
            mx.clear_cache()
    x = _linear_chunked(
        norm(x, eps=1e-5), w["output_layer.weight"], w["output_layer.bias"]
    )
    if texture:
        x = x * 0.5 + 0.5
    mx.eval(coords, x)
    if not bool(mx.all(mx.isfinite(x)).item()):
        raise FloatingPointError(f"{name} produced nonfinite output")
    return coords, x, tuple(subdivisions)
