"""Numerical checks against the repository's original PyTorch implementations."""

import importlib
import ast
import math
from pathlib import Path
import sys
import types

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from uniforge3d_mlx.ops import resize_pos_embed, packed_cross_attention
from uniforge3d_mlx.vggt import transformer_block, adapt_features
from uniforge3d_mlx.flow import dino_delta, dit_to_vggt, routed_cross_attention
from uniforge3d_mlx.sampling import sample_euler
from uniforge3d_mlx.routing import route_views

ROOT = Path(__file__).resolve().parents[1]


def original(name):
    """Load torch-only modules without executing the CUDA package initializers."""
    for package, relative in (
        ("uniforge3d", "uniforge3d"),
        ("uniforge3d.models", "uniforge3d/models"),
        ("uniforge3d.pipelines", "uniforge3d/pipelines"),
    ):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(ROOT / relative)]
            sys.modules[package] = module
    return importlib.import_module(name)


def weights(model, prefix=""):
    return {
        prefix + key: mx.array(value.detach().float().numpy())
        for key, value in model.state_dict().items()
    }


def assert_close(actual, expected, atol=2e-4, rtol=2e-4):
    np.testing.assert_allclose(
        np.array(actual), expected.detach().numpy(), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("source,target", [(37, 32), (37, 16), (8, 13)])
def test_antialiased_dino_position_resize(source, target):
    torch.manual_seed(2)
    x = torch.randn(1, source * source + 1, 12)
    patches = x[:, 1:].reshape(1, source, source, 12).permute(0, 3, 1, 2)
    ref = F.interpolate(patches, size=(target, target), mode="bicubic", antialias=True)
    ref = torch.cat((x[:, :1], ref.permute(0, 2, 3, 1).reshape(1, -1, 12)), 1)
    assert_close(
        resize_pos_embed(mx.array(x.numpy()), target, target), ref, atol=4e-6, rtol=1e-5
    )


@pytest.mark.parametrize("rope", [False, True])
def test_vggt_block_matches_source(rope):
    block = original("uniforge3d.models.vggt.layers.block")
    rope_module = original("uniforge3d.models.vggt.layers.rope")
    torch.manual_seed(2)
    rotation = rope_module.RotaryPositionEmbedding2D(frequency=100) if rope else None
    model = block.Block(32, 4, init_values=0.1, qk_norm=rope, rope=rotation).eval()
    x = torch.randn(2, 13, 32)
    positions = torch.randint(0, 10, (2, 13, 2)) if rope else None
    with torch.no_grad():
        expected = model(x, pos=positions)
    actual = transformer_block(
        mx.array(x.numpy()),
        weights(model, "b."),
        "b",
        num_heads=4,
        positions=mx.array(positions.numpy()) if rope else None,
    )
    assert_close(actual, expected)


def test_shape_attention_matches_source():
    module = original("uniforge3d.models.vggt_adapter")
    torch.manual_seed(3)
    model = module.PackedCrossAttention(32, 48, 32, 4).eval()
    query, context = torch.randn(1, 17, 32), torch.randn(23, 48)
    with torch.no_grad():
        expected = model(query, context, torch.tensor([0, 23]))
    actual = packed_cross_attention(
        mx.array(query.numpy()),
        mx.array(context.numpy())[None],
        weights(model, "a."),
        "a",
        num_heads=4,
    )
    assert_close(actual, expected)


def test_four_layer_adapter_matches_source():
    module = original("uniforge3d.models.vggt_adapter")
    torch.manual_seed(4)
    model = module.VGGT4LayerDINOAdapter(
        feature_dim=12,
        per_layer_dim=8,
        hidden_dim=32,
        output_dim=8,
        input_patch_grid_size=32,
    ).eval()
    # Initialization zeros the last layer; use nonzero parameters to test the path.
    torch.nn.init.normal_(model.output_fc2.weight, std=0.1)
    cached = {layer: torch.randn(1, 2, 1029, 12) for layer in (4, 11, 17, 23)}
    source = [cached.get(i) for i in range(24)]
    with torch.no_grad():
        expected = model(source, 5)
    actual = adapt_features(
        {k: mx.array(v.numpy()) for k, v in cached.items()}, weights(model)
    )
    assert_close(actual, expected)


def test_routed_attention_uses_the_assigned_view():
    torch.manual_seed(5)
    n, c, context_dim, views, tokens, h = 17, 32, 24, 3, 11, 4
    source = {
        "to_q": torch.nn.Linear(c, c),
        "to_kv": torch.nn.Linear(context_dim, 2 * c),
        "to_out": torch.nn.Linear(c, c),
    }
    w = {
        key: value
        for name, model in source.items()
        for key, value in weights(model, "a." + name + ".").items()
    }
    x, bank = torch.randn(n, c), torch.randn(1, views, tokens, context_dim)
    selected = np.arange(n) % views
    expected = torch.empty_like(x)
    with torch.no_grad():
        q = source["to_q"](x).reshape(n, h, -1)
        for view in range(views):
            ids = np.flatnonzero(selected == view)
            k, v = source["to_kv"](bank[:, view]).chunk(2, -1)
            attended = F.scaled_dot_product_attention(
                q[ids].transpose(0, 1)[None],
                k.reshape(1, tokens, h, -1).transpose(1, 2),
                v.reshape(1, tokens, h, -1).transpose(1, 2),
            )
            expected[ids] = source["to_out"](
                attended.transpose(1, 2).reshape(len(ids), c)
            )
    actual = routed_cross_attention(
        mx.array(x.numpy()), mx.array(bank.numpy()), selected, w, "a", num_heads=h
    )
    assert_close(actual, expected)


def test_euler_guidance_matches_source():
    module = original("uniforge3d.pipelines.samplers.flow_euler")
    source = module.FlowEulerSampler(sigma_min=1e-5)
    params = dict(
        steps=7,
        rescale_t=5,
        guidance_strength=7.5,
        guidance_interval=(0.6, 1),
        guidance_rescale=0.7,
        guidance_as_difference=True,
    )
    torch.manual_seed(6)
    noise = torch.randn(1, 2, 3, 3, 3)

    def predict(x, t, conditioned):
        return x * 0.3 + t * 0.2 + (0.4 if conditioned else -0.1)

    expected = source.sample(noise, predict, **params, verbose=False).samples
    actual = sample_euler(
        mx.array(noise.numpy()),
        predict,
        {"args": {"sigma_min": 1e-5}, "params": params},
    )
    assert_close(actual, expected, atol=3e-6, rtol=1e-5)


def fusion_classes():
    # These source classes are torch-only; omit imports of the CUDA DiT model.
    source = ROOT / "uniforge3d/models/joint_dit_vggt_stage1.py"
    tree = ast.parse(source.read_text())
    names = {"_square_grid_size", "BlockDinoCondDeltaMLP", "DiTToVGGTResidual"}
    module = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def test_joint_dino_fusion_matches_source():
    torch.manual_seed(7)
    model = fusion_classes()["BlockDinoCondDeltaMLP"](dim=16, hidden_dim=32).eval()
    torch.nn.init.normal_(model.net[-1].weight, std=0.1)
    bank, aligned = torch.randn(1, 2, 21, 16), torch.randn(2, 32, 4, 4)
    with torch.no_grad():
        expected = model(bank, aligned)
    actual = dino_delta(
        mx.array(bank.numpy()), mx.array(aligned.numpy()), weights(model, "a."), "a"
    )
    assert_close(actual, expected)


def test_joint_dit_to_vggt_matches_source():
    torch.manual_seed(8)
    model = fusion_classes()["DiTToVGGTResidual"](32, 48, 48, heads=8).eval()
    torch.nn.init.normal_(model.t_mod[-1].weight, std=0.1)
    tokens, dit, time = (
        torch.randn(1, 17, 32),
        torch.randn(1, 13, 48),
        torch.randn(1, 48),
    )
    with torch.no_grad():
        expected = model(tokens, dit, time)
    actual = dit_to_vggt(
        mx.array(tokens.numpy()),
        mx.array(dit.numpy()),
        mx.array(time.numpy()),
        weights(model, "a."),
        "a",
    )
    assert_close(actual, expected)


@pytest.mark.parametrize("grid", [32, 64])
def test_router_matches_original_connected_components_and_neighbors(grid):
    module = original("uniforge3d.models.view_router")
    rng = np.random.default_rng(12)
    points = rng.uniform(-0.52, 0.52, (1, 3, 448, 448, 3)).astype(np.float32)
    confidence = rng.integers(1, 10, (1, 3, 448, 448)).astype(np.float32)
    mask = rng.random((1, 3, 448, 448)) < 0.03
    coords = np.pad(rng.integers(0, grid, (2000, 3), dtype=np.int32), ((0, 0), (1, 0)))
    source = module.GPUVoxelViewRouter(grid_resolution=grid, patch_grid_size=grid)
    with torch.no_grad():
        result = source(
            torch.from_numpy(points),
            torch.from_numpy(confidence),
            torch.from_numpy(mask),
            torch.from_numpy(coords).long(),
        )["selected_view"]
    expected = result[tuple(torch.from_numpy(coords).long().T)].numpy()
    actual = route_views(points, confidence, mask, coords, grid=grid)
    np.testing.assert_array_equal(actual, expected)


def test_dit_block_rope_modulation_and_rms_norm_match_source():
    module = original("uniforge3d.modules.transformer.modulated")
    original("uniforge3d.modules.attention.config").set_backend("sdpa")
    rope = original("uniforge3d.modules.attention.rope")
    from uniforge3d_mlx.flow import flow_config
    from mlx_spatial.trellis2_sparse_structure import _sparse_structure_block_forward

    torch.manual_seed(17)
    model = module.ModulatedTransformerCrossBlock(
        48,
        24,
        num_heads=4,
        share_mod=True,
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
    ).eval()
    x, cond, mod = torch.randn(1, 8, 48), torch.randn(1, 13, 24), torch.randn(1, 6 * 48)
    coords = torch.stack(
        torch.meshgrid(*(torch.arange(2) for _ in range(3)), indexing="ij"), -1
    ).reshape(-1, 3)
    phases = rope.RotaryPositionEmbedder(12)(coords)
    with torch.no_grad():
        expected = model(x, mod, cond, phases)
    config = flow_config(
        dict(
            resolution=2,
            in_channels=8,
            out_channels=8,
            model_channels=48,
            cond_channels=24,
            num_blocks=1,
            num_heads=4,
            mlp_ratio=4,
            pe_mode="rope",
            share_mod=True,
            qk_rms_norm=True,
            qk_rms_norm_cross=True,
        ),
        sparse=True,
    )
    actual = _sparse_structure_block_forward(
        mx.array(x.numpy()),
        mx.array(cond.numpy()),
        config,
        weights(model, "blocks.0."),
        mod=mx.array(mod.numpy()),
        block_index=0,
    )
    assert_close(actual, expected)


def test_occupancy_decoder_matches_original():
    module = original("uniforge3d.models.vaes.sparse_structure")
    from mlx_spatial.trellis2_sparse_structure import (
        SparseStructureDecoderConfig,
        _decode_sparse_structure_latent,
    )

    torch.manual_seed(19)
    model = module.SparseStructureDecoder(
        1, 2, 1, [16, 8], num_res_blocks_middle=1
    ).eval()
    for child in model.modules():
        if isinstance(child, torch.nn.Conv3d):
            torch.nn.init.normal_(child.weight, std=0.04)
    x = torch.randn(1, 2, 3, 3, 3)
    with torch.no_grad():
        expected = model(x)
    config = SparseStructureDecoderConfig(
        "SparseStructureDecoder", 1, 2, 1, (16, 8), 1, "layer", False
    )
    actual = _decode_sparse_structure_latent(
        mx.array(x.numpy()), config, weights(model)
    )
    assert_close(actual, expected)
