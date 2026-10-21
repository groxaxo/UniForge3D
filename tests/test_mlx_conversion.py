from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from uniforge3d_mlx.checkpoint import linear, load_checkpoint, read_manifest
from uniforge3d_mlx.quantization import convert_component, eligible
from uniforge3d_mlx.source import TorchSource, merge_lora


@pytest.mark.parametrize("kind", ["linear", "conv", "transpose_conv"])
def test_lora_merge_matches_separate_inference_operations(kind):
    torch.manual_seed(13)
    rank, scale = 2, 2.0
    if kind == "linear":
        base, a, b = torch.randn(5, 7), torch.randn(rank, 7), torch.randn(5, rank)
        x = torch.randn(3, 7)
        expected = F.linear(x, base) + scale * F.linear(F.linear(x, a), b)
        actual = F.linear(x, merge_lora(base, a, b, scale))
    elif kind == "conv":
        base, a, b = torch.randn(5, 3, 3, 3), torch.randn(rank, 3, 3, 3), torch.randn(5, rank, 1, 1)
        x = torch.randn(1, 3, 8, 8)
        expected = F.conv2d(x, base, padding=1) + scale * F.conv2d(F.conv2d(x, a, padding=1), b)
        actual = F.conv2d(x, merge_lora(base, a, b, scale), padding=1)
    else:
        base, a, b = torch.randn(3, 5, 4, 4), torch.randn(3, rank, 4, 4), torch.randn(5, rank, 1, 1)
        x = torch.randn(1, 3, 4, 4)
        expected = F.conv_transpose2d(x, base, stride=2) + scale * F.conv2d(F.conv_transpose2d(x, a, stride=2), b)
        actual = F.conv_transpose2d(x, merge_lora(base, a, b, scale, transpose_conv=True), stride=2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_peft_normalization_merges_updates_and_rejects_missing_pairs():
    source = TorchSource([{
        "blocks.0.self_attn.to_qkv.base_layer.weight": torch.eye(64),
        "blocks.0.self_attn.to_qkv.lora_A.default.weight": torch.ones(2, 64),
        "blocks.0.self_attn.to_qkv.lora_B.default.weight": torch.ones(64, 2),
    }])
    assert list(source) == ["blocks.0.self_attn.to_qkv.weight"]
    torch.testing.assert_close(source.tensor(list(source)[0]), torch.eye(64) + 4)
    with pytest.raises(ValueError, match="Incomplete LoRA"):
        TorchSource([{"x.weight": torch.eye(4), "x.lora_A": torch.ones(2, 4)}])


def test_only_supported_internal_matrices_are_selected():
    assert eligible("shape512", "blocks.12.self_attn.to_qkv.weight", (4608, 1536), "float32")
    assert eligible("structure_vggt", "aggregator.global_blocks.2.mlp.fc2.weight", (1024, 4096), "float32")
    assert eligible("dinov3", "layer.4.attention.q_proj.weight", (1024, 1024), "float32")
    assert eligible("rmbg", "bb.layers.2.blocks.5.attn.qkv.weight", (768, 256), "float32")
    for component, name, shape in [
        ("shape512", "out_layer.weight", (32, 1536)),
        ("shape512", "adaLN_modulation.1.weight", (9216, 1536)),
        ("shape512", "blocks.0.self_attn.to_qkv.weight", (4608, 63)),
        ("structure_vggt", "point_head.projects.0.weight", (256, 2048, 1, 1)),
        ("vggt_conditioner", "backbone_shape_adapters.4.to_q.weight", (1024, 1024)),
        ("shape_decoder", "blocks.0.0.mlp.0.weight", (4096, 1024)),
    ]:
        assert not eligible(component, name, shape, "float32")


def test_real_omlx_packed_checkpoint_roundtrip_preserves_sensitive_tensors(tmp_path):
    torch.manual_seed(7)
    head = torch.tensor([[1.0000001, 1.0000002], [3.0000002, 5.0000005]], dtype=torch.float32)
    source = TorchSource([{
        "blocks.0.mlp.mlp.0.weight": torch.randn(64, 128),
        "blocks.0.mlp.mlp.0.bias": torch.randn(64),
        "out_layer.weight": head,
        "float_scalar": torch.tensor(1.0000001, dtype=torch.float32),
        "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
    }])
    manifest = convert_component(source, "shape512", tmp_path,
                                 omlx_root=Path("/Users/op/AI/omlx"), max_shard_bytes=10000)
    weights = load_checkpoint(tmp_path / "shape512")
    assert manifest["matrix_counts"]["5"] == 1
    np.testing.assert_array_equal(np.asarray(weights["out_layer.weight"]), head.numpy())
    packed = weights["blocks.0.mlp.mlp.0.weight"]
    x = mx.array(np.random.default_rng(2).normal(size=(3, 128)).astype(np.float32))
    dense = mx.dequantize(packed.qweight, packed.scales, packed.biases, group_size=64, bits=packed.bits)
    actual = linear(x, weights, "blocks.0.mlp.mlp.0")
    expected = np.asarray(x).astype(np.float64) @ np.asarray(dense).astype(np.float64).T
    expected += np.asarray(weights["blocks.0.mlp.mlp.0.bias"])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)
    assert manifest["tensors"]["blocks.0.mlp.mlp.0.weight"]["sample_relative_mse_output"] < 0.01
    with pytest.raises(FileExistsError):
        convert_component(source, "shape512", tmp_path, omlx_root=Path("/Users/op/AI/omlx"))
    assert read_manifest(tmp_path / "shape512")["tensor_count"] == 5


def test_model_specific_allowlist_preserves_attention_and_rejects_unsupported_tensors(tmp_path):
    from uniforge3d_mlx.quantization import import_omlx, make_plan
    source = TorchSource([{
        "layer.0.attention.q_proj.weight": torch.randn(64, 64),
        "layer.0.mlp.up_proj.weight": torch.randn(128, 64),
        "norm.weight": torch.ones(64),
    }])
    source.quantization_allowlist = {"layer.0.mlp.up_proj.weight"}
    source.quantization_policy = "vision-mlp-only-source-precision-attention-v1"
    manifest = convert_component(source, "dinov3", tmp_path, omlx_root=Path("/Users/op/AI/omlx"))
    assert "quantization" not in manifest["tensors"]["layer.0.attention.q_proj.weight"]
    assert "quantization" in manifest["tensors"]["layer.0.mlp.up_proj.weight"]
    source.quantization_allowlist = {"norm.weight"}
    with pytest.raises(ValueError, match="unsupported matrices"):
        make_plan(source, "dinov3", import_omlx(Path("/Users/op/AI/omlx")))
