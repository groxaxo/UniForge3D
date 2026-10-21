"""Joint sparse-structure and routed shape/texture flow models."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_spatial.trellis2_sparse_structure import (
    SparseStructureFlowConfig,
    _multi_head_rms_norm,
    _sparse_structure_block_forward,
    _timestep_embedding,
    _gelu_tanh,
)
from mlx_spatial.trellis2_slat import (
    SLatFlowConfig,
    _slat_self_attention,
    _slat_mlp,
    _slat_rope_cos_sin,
    _apply_slat_rope,
)
from mlx_spatial.trellis2_quantization import trellis2_linear, Trellis2QuantizedMatrix

from .checkpoint import linear, load_checkpoint
from .ops import attention, conv2d, gelu, group_norm, heads, norm, silu
from .vggt import VGGT


def flow_config(args, *, sparse=False):
    cls = SparseStructureFlowConfig if sparse else SLatFlowConfig
    values = {
        "name": "UniForgeFlow",
        "initialization": "vanilla",
        "dtype": "float32",
        **args,
    }
    return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})


def timestep(weights, t):
    x = _timestep_embedding(mx.array([1000.0 * t], mx.float32), 256)
    raw = linear(
        silu(linear(x, weights, "t_embedder.mlp.0")), weights, "t_embedder.mlp.2"
    )
    return raw, linear(silu(raw), weights, "adaLN_modulation.1")


def dino_delta(bank, aligned, weights, prefix):
    x = group_norm(aligned, weights, prefix + ".net.0")
    x = gelu(conv2d(x, weights, prefix + ".net.1"))
    x = gelu(conv2d(x, weights, prefix + ".net.3", padding=1, groups=x.shape[1]))
    x = conv2d(x, weights, prefix + ".net.5").transpose(0, 2, 3, 1)
    x = x.reshape(bank.shape[0], bank.shape[1], -1, bank.shape[-1])
    return mx.concatenate((bank[:, :, :5], bank[:, :, 5:] + x), axis=2)


def dit_to_vggt(tokens, dit, t_raw, weights, prefix, gate_cap=1.0):
    q = norm(tokens, weights, prefix + ".norm_q")
    context = norm(dit, weights, prefix + ".norm_ctx")
    shift, scale, gate = mx.split(
        linear(silu(t_raw), weights, prefix + ".t_mod.1"), 3, -1
    )
    q = q * (1 + scale[:, None]) + shift[:, None]
    bias = mx.split(weights[prefix + ".attn.in_proj_bias"], 3)
    q = trellis2_linear(q, weights[prefix + ".attn.q_proj_weight"], bias[0])
    k = trellis2_linear(context, weights[prefix + ".attn.k_proj_weight"], bias[1])
    v = trellis2_linear(context, weights[prefix + ".attn.v_proj_weight"], bias[2])
    x = attention(heads(q, 8), heads(k, 8), heads(v, 8))
    x = linear(
        linear(x, weights, prefix + ".attn.out_proj"), weights, prefix + ".delta_proj"
    )
    return tokens + gate_cap * gate[:, None] * x


class JointStructureFlow:
    def __init__(self, bundle, model_args, images, bank):
        root = Path(bundle)
        self.config = flow_config(model_args["dit_args"], sparse=True)
        self.weights = load_checkpoint(root / "structure_dit")
        self.fusion = load_checkpoint(root / "structure_fusion")
        self.vggt = VGGT(root / "structure_vggt")
        self.images = images
        self.bank = bank.astype(mx.float32)
        self.patch = self.vggt.patch_tokens(images)
        mx.eval(self.patch)
        self.stages = dict(
            zip(model_args["fusion_vggt_layers"], model_args["fusion_dit_blocks"])
        )
        starts = list(self.stages.values())
        self.ends = dict(zip(starts, starts[1:] + [self.config.num_blocks]))
        self.gate_cap = model_args.get("dit_to_vggt_gate_cap", 1.0)
        self.anchor = model_args.get("anchor_view", 0)

    def __call__(self, sample, t, conditioned):
        w, cfg = self.weights, self.config
        x = sample.reshape(1, cfg.in_channels, -1).transpose(0, 2, 1)
        x = linear(x, w, "input_layer")
        t_raw, mod = timestep(w, t)

        def block(value, condition, index):
            value = _sparse_structure_block_forward(
                value, condition, cfg, w, mod=mod, block_index=index
            )
            mx.eval(value)
            return value

        if not conditioned:
            condition = mx.zeros_like(self.bank[:, self.anchor])
            for i in range(cfg.num_blocks):
                x = block(x, condition, i)
        else:
            tokens, positions = self.vggt.initialize(self.images, self.patch)
            views = self.images.shape[1]
            anchor = self.bank[:, self.anchor]
            next_block = 0
            for layer in range(24):
                frame, global_value = self.vggt.layer(tokens, positions, layer, views)
                if layer in self.stages:
                    start = self.stages[layer]
                    while next_block < start:
                        x = block(x, anchor, next_block)
                        next_block += 1
                    maps = [
                        value[:, :, 5:]
                        .reshape(views, 32, 32, 1024)
                        .transpose(0, 3, 1, 2)
                        for value in (frame, global_value)
                    ]
                    aligned = mx.concatenate(maps, axis=1)
                    for i in range(start, self.ends[start]):
                        bank = dino_delta(
                            self.bank, aligned, self.fusion, f"cond_delta_mlps.{i}"
                        )
                        x = block(x, bank.reshape(1, -1, 1024), i)
                        next_block += 1
                    global_value = dit_to_vggt(
                        global_value.reshape(1, -1, 1024),
                        x,
                        t_raw,
                        self.fusion,
                        f"dit_to_vggt.{start}",
                        self.gate_cap,
                    ).reshape(1, views, -1, 1024)
                tokens = global_value.reshape(views, -1, 1024)
                mx.eval(tokens)
            while next_block < cfg.num_blocks:
                x = block(x, anchor, next_block)
                next_block += 1
        x = linear(norm(x, eps=1e-5), w, "out_layer")
        return x.transpose(0, 2, 1).reshape(sample.shape)


def mixed_linear(x, weights, prefix, compute_dtype=mx.float32):
    """FP32 residuals and biases, optionally with FP16 matrix multiplication."""
    if compute_dtype == mx.float32:
        return linear(x, weights, prefix)
    value = trellis2_linear(x.astype(compute_dtype), weights[prefix + ".weight"], None)
    value = value.astype(mx.float32)
    bias = weights.get(prefix + ".bias")
    return value if bias is None else value + bias.astype(mx.float32)


def routed_cross_attention(
    x, bank, selected, weights, prefix, *, num_heads=12, compute_dtype=mx.float32
):
    """Group queries by selected view, retaining all image tokens in that view."""
    q = mixed_linear(x, weights, prefix + ".to_q", compute_dtype)[None]
    head_dim = q.shape[-1] // num_heads
    q = q.reshape(1, -1, num_heads, head_dim)
    if prefix + ".q_rms_norm.gamma" in weights:
        q = _multi_head_rms_norm(
            q, weights[prefix + ".q_rms_norm.gamma"].astype(mx.float32)
        )
    q = q.transpose(0, 2, 1, 3)
    output = mx.zeros((x.shape[0], num_heads * head_dim), dtype=x.dtype)
    for view in np.unique(selected):
        ids = mx.array(np.flatnonzero(selected == view).astype(np.int32))
        kv = mixed_linear(bank[:, int(view)], weights, prefix + ".to_kv", compute_dtype)
        k, v = mx.split(kv, 2, axis=-1)
        k = k.reshape(1, -1, num_heads, head_dim)
        if prefix + ".k_rms_norm.gamma" in weights:
            k = _multi_head_rms_norm(
                k, weights[prefix + ".k_rms_norm.gamma"].astype(mx.float32)
            )
        y = attention(
            q[:, :, ids].astype(compute_dtype),
            k.transpose(0, 2, 1, 3).astype(compute_dtype),
            heads(v, num_heads).astype(compute_dtype),
        )[0].astype(mx.float32)
        output = output.at[ids].add(y)
    return mixed_linear(output, weights, prefix + ".to_out", compute_dtype)


class RoutedFlow:
    def __init__(
        self,
        component,
        model_args,
        coordinates,
        bank,
        selected,
        concat_features=None,
        compute_dtype="float32",
    ):
        self.weights = load_checkpoint(component)
        self.config = flow_config(model_args)
        self.routed_layers = set(model_args["routed_layers"])
        self.coords = mx.array(coordinates, dtype=mx.int32)
        self.selected = np.asarray(selected, dtype=np.int32)
        self.bank = bank.astype(mx.float32)
        self.concat = concat_features
        if compute_dtype not in ("float32", "mixed"):
            raise ValueError("Flow compute dtype must be float32 or mixed")
        self.compute_dtype = mx.float16 if compute_dtype == "mixed" else mx.float32
        if self.compute_dtype == mx.float16:
            self.weights = {
                key: replace(
                    value,
                    scales=value.scales.astype(mx.float16),
                    biases=value.biases.astype(mx.float16),
                )
                if isinstance(value, Trellis2QuantizedMatrix)
                else value
                for key, value in self.weights.items()
            }
        self.rope = _slat_rope_cos_sin(self.coords, self.config)
        if self.selected.shape != (len(coordinates),):
            raise ValueError("Routing must have one view index per sparse token")
        if self.selected.min() < 0 or self.selected.max() >= bank.shape[1]:
            raise ValueError("View assignment is outside the input view bank")

    def __call__(self, sample, t, conditioned):
        w, cfg = self.weights, self.config
        input_features = (
            sample if self.concat is None else mx.concatenate((sample, self.concat), -1)
        )
        x = linear(input_features, w, "input_layer")
        _, mod = timestep(w, t)
        bank = self.bank if conditioned else mx.zeros_like(self.bank)
        selected = self.selected if conditioned else np.zeros_like(self.selected)
        all_views = bank.reshape(1, 1, -1, bank.shape[-1])
        for i in range(cfg.num_blocks):
            prefix = f"blocks.{i}"
            shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = mx.split(
                w[prefix + ".modulation"] + mod, 6, -1
            )
            h = norm(x, eps=1e-6) * (1 + scale) + shift
            x = x + self.self_attention(h, i) * gate
            h = norm(x, w, prefix + ".norm2", eps=1e-6)
            routed_bank = bank if i in self.routed_layers else all_views
            route = selected if i in self.routed_layers else np.zeros_like(selected)
            x = x + routed_cross_attention(
                h,
                routed_bank,
                route,
                w,
                prefix + ".cross_attn",
                num_heads=cfg.num_heads,
                compute_dtype=self.compute_dtype,
            )
            h = norm(x, eps=1e-6) * (1 + scale_mlp) + shift_mlp
            if self.compute_dtype == mx.float32:
                mlp = _slat_mlp(h, w, block_index=i)
            else:
                mlp = mixed_linear(h, w, prefix + ".mlp.mlp.0", self.compute_dtype)
                mlp = mixed_linear(
                    _gelu_tanh(mlp), w, prefix + ".mlp.mlp.2", self.compute_dtype
                )
            x = x + mlp * gate_mlp
            mx.eval(x)
        return linear(norm(x, eps=1e-5), w, "out_layer")

    def self_attention(self, x, index):
        if self.compute_dtype == mx.float32:
            return _slat_self_attention(
                x, self.coords, self.config, self.weights, block_index=index
            )
        w, cfg = self.weights, self.config
        prefix = f"blocks.{index}.self_attn"
        qkv = mixed_linear(x, w, prefix + ".to_qkv", self.compute_dtype)
        qkv = qkv.reshape(
            1, len(x), 3, cfg.num_heads, cfg.model_channels // cfg.num_heads
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if cfg.qk_rms_norm:
            q = _multi_head_rms_norm(
                q, w[prefix + ".q_rms_norm.gamma"].astype(mx.float32)
            )
            k = _multi_head_rms_norm(
                k, w[prefix + ".k_rms_norm.gamma"].astype(mx.float32)
            )
        if cfg.pe_mode == "rope":
            q, k = _apply_slat_rope(q, *self.rope), _apply_slat_rope(k, *self.rope)
        value = attention(
            q.transpose(0, 2, 1, 3).astype(mx.float16),
            k.transpose(0, 2, 1, 3).astype(mx.float16),
            v.transpose(0, 2, 1, 3).astype(mx.float16),
        )[0].astype(mx.float32)
        return mixed_linear(value, w, prefix + ".to_out", self.compute_dtype)
