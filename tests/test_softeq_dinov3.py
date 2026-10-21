import pytest
import torch
from torch.nn import functional as F

from uniforge3d_mlx.softeq import fold_patch_kernel, normalize_softeq_state


@pytest.mark.parametrize("preserve_norm", [False, True])
def test_patch_filter_fold_matches_published_per_channel_operation(preserve_norm):
    torch.manual_seed(11)
    weight = torch.randn(7, 3, 4, 4)
    # An asymmetric matrix catches the otherwise easy-to-miss transpose error.
    projection = torch.randn(16, 16)
    expected = weight.flatten(-2) @ projection.T
    if preserve_norm:
        expected *= (weight.flatten(-2).norm(dim=-1) / (expected.norm(dim=-1) + 1e-8)).unsqueeze(-1)
    actual = fold_patch_kernel(weight, projection, softness=0.8, preserve_norm=preserve_norm)
    x = torch.randn(1, 3, 32, 32)
    torch.testing.assert_close(F.conv2d(x, actual, stride=4),
                               F.conv2d(x, expected.reshape(weight.shape), stride=4))


def test_identity_patch_filter_shortcut_does_not_apply_stored_buffer():
    weight = torch.randn(5, 3, 4, 4)
    assert torch.equal(fold_patch_kernel(weight, torch.zeros(16, 16), softness=1.0), weight)


def test_normalizer_refuses_to_silently_drop_positional_filtering():
    with pytest.raises(ValueError, match="positional filtering"):
        normalize_softeq_state({}, {"model_arch": "filtered_dinov3_seg", "soft_thresholding_pos": 0.8})


def test_normalizer_requires_all_published_buffers_and_rejects_unknown_filtering():
    config = {"model_arch": "filtered_dinov3_seg", "soft_thresholding_pos": 1.0,
              "filter_patch_embeddings": True, "soft_thresholding": 0.8}
    with pytest.raises(ValueError, match="Missing source filter buffer"):
        normalize_softeq_state({}, config)
    with pytest.raises(ValueError, match="Attention or MLP"):
        normalize_softeq_state({}, {**config, "filter_mlp": True})
