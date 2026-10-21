"""Convert the explicitly selected SoftEq backbone without executing remote code."""
from __future__ import annotations

import torch

SOFTEQ_REPO = "ashiq24/softeq-dinov3-vitl16-pretrain-lvd1689m-ade-seg-c360-s0.1-sp0.8"
SOFTEQ_REVISION = "f853065890a0df7869f9b5267361cdc0ac139fee"
SOFTEQ_SHA256 = "c2c6df12b078eecbf00cea076fabdae803dafa01bf53f26648ebdcb01eb0f905"
BACKBONE_CONFIG = "backbone_config_dinov3-vitl16-pretrain-lvd1689m.json"


def fold_patch_kernel(weight, projection, *, softness, preserve_norm=False):
    """Fold InvariantProjector.forward exactly into a Conv2d spatial kernel.

    The published forward uses einsum('fi,bci->bcf', filter_w_T, W),
    independently for each RGB channel. The matrix must not be transposed.
    """
    if weight.ndim != 4 or projection.shape != (weight.shape[-2] * weight.shape[-1],) * 2:
        raise ValueError("Patch filter shape does not match the convolution kernel")
    if softness == 1.0:
        return weight.clone()
    flattened = weight.reshape(weight.shape[0], weight.shape[1], -1)
    filtered = torch.einsum("fi,bci->bcf", projection, flattened)
    if preserve_norm:
        filtered = filtered * (flattened.norm(dim=-1) / (filtered.norm(dim=-1) + 1e-8)).unsqueeze(-1)
    return filtered.reshape(weight.shape)


def normalize_softeq_state(state, config):
    """Extract a feature backbone, rejecting unimplemented behavior changes."""
    if config.get("model_arch") != "filtered_dinov3_seg":
        raise ValueError("Expected the selected filtered DINOv3 segmentation architecture")
    if config.get("soft_thresholding_pos") != 1.0:
        raise ValueError("Non-identity positional filtering cannot be dropped")
    if any(config.get(k, False) for k in ("filter_attention_qkv", "filter_attention_output", "filter_mlp")):
        raise ValueError("Attention or MLP filtering needs a separate conversion")
    if not config.get("filter_patch_embeddings"):
        raise ValueError("Expected a filtered patch embedding in this checkpoint")
    names = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    filter_keys = (
        "embeddings.patch_embeddings.filter.filter_w_T",
        "embeddings.patch_embeddings.filter.filter_x",
        "rope_embeddings.filter_coords.filter_w_T",
        "rope_embeddings.filter_coords.filter_x",
    )
    for key in filter_keys:
        if key not in names:
            raise ValueError(f"Missing source filter buffer: {key}")
    patch = "embeddings.patch_embeddings.weight"
    names[patch] = fold_patch_kernel(names[patch], names[filter_keys[0]],
                                    softness=config["soft_thresholding"],
                                    preserve_norm=config.get("preserve_norm", False))
    for key in filter_keys:
        del names[key]
    extra = {k for k in state if not k.startswith("backbone.")}
    if extra != {"classifier.weight", "classifier.bias"}:
        raise ValueError(f"Unexpected non-backbone parameters: {extra}")
    return names


def torch_backbone(config, canonical_state, *, filtered_reference_state=None, soft_config=None):
    """Instantiate the installed HF implementation with strict key/shape checking."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig.from_dict(config)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = DINOv3ViTModel(cfg)
    expected = set(model.state_dict())
    renamed = {}
    for key, value in canonical_state.items():
        target = key if key in expected else "model." + key
        if target not in expected:
            raise ValueError(f"No HF parameter for normalized backbone tensor: {key}")
        renamed[target] = value
    model.load_state_dict(renamed, strict=True, assign=True)
    # RoPE frequencies are non-persistent and therefore absent from state_dict.
    # Recreate this deterministic buffer outside the meta construction context.
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
    model.rope_embeddings = DINOv3ViTRopePositionEmbedding(cfg)
    if filtered_reference_state is not None:
        # A small local implementation of the reviewed published Conv2d wrapper.
        # The complete HF model still runs locally; no trust_remote_code load.
        class PublishedFilteredPatch(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.weight = torch.nn.Parameter(filtered_reference_state["backbone.embeddings.patch_embeddings.weight"])
                self.bias = torch.nn.Parameter(filtered_reference_state["backbone.embeddings.patch_embeddings.bias"])
                self.register_buffer("projection", filtered_reference_state["backbone.embeddings.patch_embeddings.filter.filter_w_T"])
                self.stride, self.padding = original.stride, original.padding

            def forward(self, x):
                shape = self.weight.shape
                w = self.weight.reshape(shape[0], -1)
                if soft_config["soft_thresholding"] != 1.0:
                    w = w.reshape(shape[0], shape[1], -1)
                    before = w.norm(dim=-1)
                    w = torch.einsum("fi,bci->bcf", self.projection, w)
                    if soft_config.get("preserve_norm", False):
                        w = w * (before / (w.norm(dim=-1) + 1e-8)).unsqueeze(-1)
                return torch.nn.functional.conv2d(x, w.reshape(shape), self.bias, stride=self.stride, padding=self.padding)

        model.embeddings.patch_embeddings = PublishedFilteredPatch(model.embeddings.patch_embeddings)
    return model.eval().requires_grad_(False)


def torch_features(model, pixels):
    """Follow UniForge3D's explicit final normalization rather than the ADE head."""
    hidden = model.embeddings(pixels, bool_masked_pos=None)
    position = model.rope_embeddings(pixels)
    if hasattr(model, "model"):
        hidden = model.model(hidden, position_embeddings=position).last_hidden_state
    else:
        for layer in model.layer:
            hidden = layer(hidden, position_embeddings=position)
    return torch.nn.functional.layer_norm(hidden, hidden.shape[-1:])
