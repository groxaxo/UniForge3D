"""UniForge's DINOv2/VGGT backbone, shape conditioning and point head in MLX."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from .checkpoint import linear, load_checkpoint, read_manifest
from .ops import (
    attention,
    conv2d,
    gelu,
    heads,
    norm,
    packed_cross_attention,
    resize_pos_embed,
    rope2d,
)

FEATURE_LAYERS = (4, 11, 17, 23)
CONDITION_LAYERS = (4, 7, 11, 14, 17, 20, 23)


def transformer_block(x, weights, prefix, *, positions=None, num_heads=16, eps=1e-5):
    qkv = linear(
        norm(x, weights, prefix + ".norm1", eps), weights, prefix + ".attn.qkv"
    )
    q, k, v = (heads(part, num_heads) for part in mx.split(qkv, 3, axis=-1))
    if prefix + ".attn.q_norm.weight" in weights:
        q = norm(q, weights, prefix + ".attn.q_norm")
        k = norm(k, weights, prefix + ".attn.k_norm")
    if positions is not None:
        q, k = rope2d(q, positions), rope2d(k, positions)
    out = linear(attention(q, k, v), weights, prefix + ".attn.proj")
    x = x + out * weights[prefix + ".ls1.gamma"].astype(x.dtype)
    mlp = linear(norm(x, weights, prefix + ".norm2", eps), weights, prefix + ".mlp.fc1")
    mlp = linear(gelu(mlp), weights, prefix + ".mlp.fc2")
    return x + mlp * weights[prefix + ".ls2.gamma"].astype(x.dtype)


class VGGT:
    def __init__(self, root: str | Path, *, conditioner=False):
        manifest = read_manifest(root)
        names = list(manifest["tensors"])
        if not conditioner:
            names = [name for name in names if name.startswith("aggregator.")]
        self.weights = load_checkpoint(root, names=names)
        self._positions = {}

    def patch_tokens(self, images):
        """Return the frozen DINOv2 patch features for [B,V,3,448,448]."""
        b, v, c, h, w = images.shape
        if b != 1 or c != 3 or (h, w) != (448, 448):
            raise ValueError("VGGT expects one case with RGB views at 448x448")
        weights = self.weights
        prefix = "aggregator.patch_embed"
        mean = mx.array([0.485, 0.456, 0.406])[None, :, None, None]
        std = mx.array([0.229, 0.224, 0.225])[None, :, None, None]
        x = (images.reshape(b * v, c, h, w).astype(mx.float32) - mean) / std
        x = conv2d(x, weights, prefix + ".patch_embed.proj", stride=14)
        x = x.transpose(0, 2, 3, 1).reshape(b * v, -1, x.shape[1])
        cls = mx.broadcast_to(weights[prefix + ".cls_token"], (b * v, 1, x.shape[-1]))
        x = mx.concatenate((cls, x), axis=1)
        pos_key = (h // 14, w // 14)
        if pos_key not in self._positions:
            self._positions[pos_key] = resize_pos_embed(
                weights[prefix + ".pos_embed"], *pos_key
            )
        x = x + self._positions[pos_key]
        registers = mx.broadcast_to(
            weights[prefix + ".register_tokens"], (b * v, 4, x.shape[-1])
        )
        x = mx.concatenate((x[:, :1], registers, x[:, 1:]), axis=1)
        for i in range(24):
            x = transformer_block(x, weights, f"{prefix}.blocks.{i}", eps=1e-6)
            mx.eval(x)
        return norm(x, weights, prefix + ".norm", eps=1e-6)[:, 5:]

    def initialize(self, images, patch_tokens=None):
        b, v = images.shape[:2]
        patch = self.patch_tokens(images) if patch_tokens is None else patch_tokens

        def special(name):
            value = self.weights["aggregator." + name].astype(patch.dtype)
            return mx.concatenate(
                (value[:, :1], mx.repeat(value[:, 1:2], v - 1, axis=1)), 1
            ).reshape(b * v, -1, patch.shape[-1])

        tokens = mx.concatenate(
            (special("camera_token"), special("register_token"), patch), 1
        )
        grid = mx.arange(32, dtype=mx.float32)
        yy, xx = mx.meshgrid(grid, grid, indexing="ij")
        positions = mx.concatenate(
            (mx.zeros((5, 2)), mx.stack((yy, xx), -1).reshape(-1, 2) + 1), 0
        )
        return tokens, mx.broadcast_to(positions, (b * v, *positions.shape))

    def layer(self, tokens, positions, layer, views):
        w = self.weights
        frame = transformer_block(
            tokens.reshape(views, -1, 1024),
            w,
            f"aggregator.frame_blocks.{layer}",
            positions=positions,
        )
        global_value = transformer_block(
            frame.reshape(1, -1, 1024),
            w,
            f"aggregator.global_blocks.{layer}",
            positions=positions.reshape(1, -1, 2),
        )
        mx.eval(frame, global_value)
        return frame.reshape(1, views, -1, 1024), global_value.reshape(
            1, views, -1, 1024
        )

    def shape_tokens(self, coords):
        xyz = (coords.astype(mx.float32) + 0.5) / 64.0 - 0.5
        phase = 2 * math.pi * xyz[..., None] * 2.0 ** mx.arange(10, dtype=mx.float32)
        x = mx.concatenate(
            (xyz, mx.sin(phase).reshape(-1, 30), mx.cos(phase).reshape(-1, 30)), -1
        )
        for name in ("coord_mlp", "token_mlp"):
            prefix = "shape_encoder." + name + ".net"
            x = linear(
                gelu(linear(x, self.weights, prefix + ".0")),
                self.weights,
                prefix + ".2",
            )
        return x[None]

    def conditioned_features(
        self, images, coords, *, patch_tokens=None, return_points=True
    ):
        tokens, positions = self.initialize(images, patch_tokens)
        shape = self.shape_tokens(coords)
        views = images.shape[1]
        cached = {}
        for i in range(24):
            frame, global_value = self.layer(tokens, positions, i, views)
            if i in CONDITION_LAYERS:
                value = packed_cross_attention(
                    global_value.reshape(1, -1, 1024),
                    shape,
                    self.weights,
                    f"backbone_shape_adapters.{i}",
                    num_heads=16,
                )
                global_value = value.reshape(global_value.shape)
            tokens = global_value.reshape(views, -1, 1024)
            if i in FEATURE_LAYERS:
                cached[i] = mx.concatenate((frame, global_value), -1)
            mx.eval(tokens, *cached.values())
        points = (
            self.point_head(cached, shape, views) if return_points else (None, None)
        )
        return cached, *points

    def point_head(self, cached, shape_tokens, views):
        # These DPT helpers implement the same in-place-ReLU residual semantics,
        # positional grid and align_corners=True interpolation as upstream VGGT.
        from mlx_spatial.hyworld2_heads import (
            _apply_dpt_pos_embed,
            _conv2d_nchw,
            _feature_fusion_block,
            _official_dpt_resize,
            _resize_nchw,
        )

        w = {
            key.removeprefix("point_head."): value.astype(mx.float32)
            for key, value in self.weights.items()
            if key.startswith("point_head.")
        }
        levels = []
        for i, layer in enumerate(FEATURE_LAYERS):
            tokens = norm(
                cached[layer][:, :, 5:].reshape(views, 1024, 2048).astype(mx.float32),
                w,
                "norm",
            )
            x = tokens.transpose(0, 2, 1).reshape(views, 2048, 32, 32)
            x = conv2d(x, w, f"projects.{i}")
            x = _apply_dpt_pos_embed(x, width=448, height=448)
            x = _official_dpt_resize(x, i, w)
            x = _conv2d_nchw(x, w[f"scratch.layer{i + 1}_rn.weight"], None)
            mx.eval(x)
            levels.append(x)
        x = _feature_fusion_block(
            levels[3],
            None,
            prefix="scratch.refinenet4",
            tensors=w,
            size=levels[2].shape[2:],
        )
        shape = x.shape
        query = x.transpose(0, 2, 3, 1).reshape(1, -1, 256)
        x = packed_cross_attention(
            query,
            shape_tokens.astype(mx.float32),
            self.weights,
            "dpt_shape_adapter_32.attn",
            num_heads=8,
        )
        x = x.reshape(views, shape[2], shape[3], 256).transpose(0, 3, 1, 2)
        for i in (3, 2, 1):
            size = levels[i - 2].shape[2:] if i > 1 else None
            x = _feature_fusion_block(
                x, levels[i - 1], prefix=f"scratch.refinenet{i}", tensors=w, size=size
            )
            mx.eval(x)
        x = conv2d(x, w, "scratch.output_conv1", padding=1)
        x = _resize_nchw(x, (448, 448))
        x = _apply_dpt_pos_embed(x, width=448, height=448)
        x = mx.maximum(conv2d(x, w, "scratch.output_conv2.0", padding=1), 0)
        raw = conv2d(x, w, "scratch.output_conv2.2").transpose(0, 2, 3, 1)[None]
        xyz = raw[..., :3]
        return mx.sign(xyz) * mx.expm1(mx.abs(xyz)), 1 + mx.exp(raw[..., 3])


def adapt_features(cached, weights, *, upsample=False):
    projected = [
        linear(
            norm(cached[layer][:, :, 5:], weights, f"input_norms.{i}"),
            weights,
            f"input_projections.{i}",
        )
        for i, layer in enumerate(FEATURE_LAYERS)
    ]
    x = mx.concatenate(projected, -1)
    x = x + linear(
        gelu(linear(norm(x, weights, "hidden_norm"), weights, "hidden_fc1")),
        weights,
        "hidden_fc2",
    )
    x = linear(
        gelu(linear(norm(x, weights, "output_norm"), weights, "output_fc1")),
        weights,
        "output_fc2",
    )
    if upsample:
        b, v, _, c = x.shape
        x = x.reshape(b * v, 32, 32, c).transpose(0, 3, 1, 2)
        x = mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)
        delta = gelu(conv2d(x, weights, "up_refine.0", padding=1, groups=c))
        x = x + conv2d(delta, weights, "up_refine.2")
        x = x.transpose(0, 2, 3, 1).reshape(b, v, 4096, c)
    return x
