"""Local MLX DINOv3 feature encoder for UniForge3D's 4D/5D tensor contract."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .checkpoint import load_checkpoint


class DinoV3Encoder:
    def __init__(self, root: str | Path, *, config_path: str | Path | None = None):
        from mlx_spatial.trellis2_dinov3 import read_dinov3_config
        root = Path(root)
        inspected = read_dinov3_config(config_path or root / "config.json")
        if inspected.blocker is not None or inspected.config is None:
            raise ValueError(inspected.blocker)
        self.config = inspected.config
        if (root / "manifest.json").is_file():
            self.weights = load_checkpoint(root)
        else:
            self.weights = mx.load(str(root / "model.safetensors"))

    def __call__(self, images):
        from mlx_spatial.trellis2_dinov3_forward import assemble_dinov3_tokens, run_dinov3_transformer_block
        images = mx.array(images).astype(mx.float32)
        multiview = images.ndim == 5
        if images.ndim not in (4, 5):
            raise ValueError("Expected DINO images [B,3,H,W] or [B,V,3,H,W]")
        if multiview:
            batch, views, channels, height, width = images.shape
            images = images.reshape(batch * views, channels, height, width)
        _, channels, height, width = images.shape
        if channels != 3 or height != width or height % self.config.patch_size:
            raise ValueError("DINO inputs must be square RGB images divisible by patch size")
        mean = mx.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = mx.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        assembled = assemble_dinov3_tokens((images - mean) / std, self.config, self.weights)
        if not assembled.ready:
            raise RuntimeError(assembled.blocker)
        hidden = assembled.tokens
        for i in range(self.config.num_hidden_layers):
            result = run_dinov3_transformer_block(hidden, self.config, self.weights,
                                                  patch_grid=assembled.patch_grid, layer_index=i)
            if not result.ready:
                raise RuntimeError(result.blocker)
            hidden = result.hidden_states
            # Bound graph lifetime/activation memory at the real 1024px tier.
            mx.eval(hidden)
        hidden = mx.fast.layer_norm(hidden, None, None, eps=1e-5)
        if multiview:
            hidden = hidden.reshape(batch, views, hidden.shape[1], hidden.shape[2])
        return hidden.astype(mx.float32)
