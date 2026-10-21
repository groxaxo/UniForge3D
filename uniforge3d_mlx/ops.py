"""Small functional MLX operators shared by UniForge's inference stages."""

from __future__ import annotations

import gc
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .checkpoint import linear


def release_memory():
    mx.synchronize()
    gc.collect()
    mx.clear_cache()


def norm(x, weights=None, prefix=None, eps=1e-5):
    weight = (
        weights[prefix + ".weight"].astype(x.dtype) if weights is not None else None
    )
    bias = weights[prefix + ".bias"].astype(x.dtype) if weights is not None else None
    return mx.fast.layer_norm(x, weight, bias, eps)


def gelu(x):
    return nn.gelu(x)


def silu(x):
    return x * mx.sigmoid(x)


def conv2d(x, weights, prefix, *, stride=1, padding=0, groups=1):
    """NCHW activations and PyTorch OIHW checkpoint weights."""
    w = weights[prefix + ".weight"].astype(x.dtype).transpose(0, 2, 3, 1)
    y = mx.conv2d(
        x.transpose(0, 2, 3, 1), w, stride=stride, padding=padding, groups=groups
    )
    if prefix + ".bias" in weights:
        y = y + weights[prefix + ".bias"].astype(x.dtype)
    return y.transpose(0, 3, 1, 2)


def group_norm(x, weights, prefix, groups=32, eps=1e-5):
    shape = x.shape
    grouped = x.astype(mx.float32).reshape(shape[0], groups, -1)
    grouped = (grouped - grouped.mean(-1, keepdims=True)) * mx.rsqrt(
        grouped.var(-1, keepdims=True) + eps
    )
    return (
        grouped.reshape(shape) * weights[prefix + ".weight"][None, :, None, None]
        + weights[prefix + ".bias"][None, :, None, None]
    ).astype(x.dtype)


def heads(x, count):
    return x.reshape(*x.shape[:-1], count, x.shape[-1] // count).transpose(0, 2, 1, 3)


def attention(q, k, v, *, query_chunk=2048):
    """Exact full attention, bounded by query chunks; BHND layout."""
    chunks = []
    for start in range(0, q.shape[2], query_chunk):
        y = mx.fast.scaled_dot_product_attention(
            q[:, :, start : start + query_chunk], k, v, scale=q.shape[-1] ** -0.5
        )
        mx.eval(y)
        chunks.append(y)
    y = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
    return y.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


def rope2d(x, positions, base=100.0):
    half = x.shape[-1] // 2
    frequency = base ** (-mx.arange(0, half, 2, dtype=mx.float32) / half)
    parts = []
    for axis, value in enumerate(mx.split(x, 2, axis=-1)):
        angle = (positions[..., axis, None] * frequency).astype(x.dtype)
        angle = mx.concatenate((angle, angle), axis=-1)[:, None]
        left, right = mx.split(value, 2, axis=-1)
        rotated = mx.concatenate((-right, left), axis=-1)
        parts.append(value * mx.cos(angle) + rotated * mx.sin(angle))
    return mx.concatenate(parts, axis=-1)


def packed_cross_attention(query, context, weights, prefix, *, num_heads):
    """One case's packed shape-token attention (batch size one)."""
    q = linear(norm(query, weights, prefix + ".q_norm"), weights, prefix + ".to_q")
    k = linear(norm(context, weights, prefix + ".k_norm"), weights, prefix + ".to_k")
    v = linear(context, weights, prefix + ".to_v")
    y = attention(heads(q, num_heads), heads(k, num_heads), heads(v, num_heads))
    return query + linear(y, weights, prefix + ".out_proj")


def resize_pos_embed(pos_embed, height, width):
    """DINOv2's PyTorch bicubic antialias=True, align_corners=False.

    Only constant positional weights are resized on the CPU. Unlike scipy.zoom,
    this uses the same truncated, renormalized antialias kernel as PyTorch.
    """
    side = math.isqrt(pos_embed.shape[1] - 1)
    if side * side != pos_embed.shape[1] - 1:
        raise ValueError("DINOv2 positional grid must be square")
    if (height, width) == (side, side):
        return pos_embed.astype(mx.float32)

    def coefficients(target):
        scale = side / target
        filter_scale = max(scale, 1.0)
        center = (np.arange(target, dtype=np.float64) + 0.5) * scale
        distance = np.abs(
            (np.arange(side)[None] + 0.5 - center[:, None]) / filter_scale
        )
        # Antialiased bicubic uses a=-0.5 (non-AA bicubic uses -0.75).
        a = -0.5
        inside = ((a + 2) * distance - (a + 3)) * distance**2 + 1
        outside = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        weights = np.where(distance < 1, inside, np.where(distance < 2, outside, 0))
        return weights / weights.sum(1, keepdims=True)

    p = np.array(pos_embed.astype(mx.float32))
    grid = p[0, 1:].reshape(side, side, -1)
    grid = np.einsum("hi,ijc,wj->hwc", coefficients(height), grid, coefficients(width))
    return mx.array(
        np.concatenate((p[:, :1], grid.reshape(1, height * width, -1)), 1),
        dtype=mx.float32,
    )
