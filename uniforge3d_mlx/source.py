"""Read released tensors lazily and merge the inference LoRA updates correctly."""
from __future__ import annotations

from collections.abc import Mapping
import re
from pathlib import Path

import torch


def load_torch_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for wrapper in ("model", "state_dict"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
            break
    return {str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def merge_lora(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float,
               *, transpose_conv: bool = False) -> torch.Tensor:
    base, a, b = base.float(), a.float(), b.float()
    if base.ndim == 2:
        update = b @ a
    elif base.ndim == 4 and transpose_conv:
        update = torch.einsum("or,irhw->iohw", b[:, :, 0, 0], a)
    elif base.ndim == 4:
        update = torch.einsum("or,rihw->oihw", b[:, :, 0, 0], a)
    else:
        raise ValueError(f"Unsupported LoRA base shape {tuple(base.shape)}")
    if base.shape != update.shape:
        raise ValueError(f"LoRA shape mismatch: {tuple(base.shape)} != {tuple(update.shape)}")
    return base + update * scale


class TorchSource:
    """A normalized tensor mapping; only requested merged weights are materialized."""
    def __init__(self, states: list[dict], *, scale: float = 2.0):
        self.scale = scale
        combined = {}
        for state in states:
            combined.update(state)
        self.original = combined
        self.aliases = {}
        self.lora = {}
        for name in combined:
            match = re.match(r"^(.*)\.lora_([AB])(?:\.default\.weight)?$", name)
            if match:
                self.lora.setdefault(match[1], {})[match[2]] = name
                continue
            normalized = name.replace(".base_layer.", ".").replace(".base.", ".")
            if normalized in self.aliases and self.aliases[normalized] != name:
                raise ValueError(f"Ambiguous base weights for {normalized}")
            self.aliases[normalized] = name
        for prefix, pair in self.lora.items():
            if set(pair) != {"A", "B"} or prefix + ".weight" not in self.aliases:
                raise ValueError(f"Incomplete LoRA update: {prefix}")

    def __iter__(self):
        return iter(sorted(self.aliases))

    def shape(self, name):
        return tuple(self.original[self.aliases[name]].shape)

    def dtype(self, name):
        return str(self.original[self.aliases[name]].dtype).removeprefix("torch.")

    def tensor(self, name):
        value = self.original[self.aliases[name]]
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix in self.lora:
            pair = self.lora[prefix]
            # These are the ConvTranspose2d modules wrapped by Stage1 point-head LoRA.
            transpose = bool(re.search(r"point_head\.resize_layers\.[01]$", prefix))
            value = merge_lora(value, self.original[pair["A"]], self.original[pair["B"]],
                               self.scale, transpose_conv=transpose)
        return value.detach()


class SafeSource:
    def __init__(self, path: Path):
        from safetensors import safe_open
        self.handle = safe_open(str(path), framework="pt", device="cpu")
        self.names = list(self.handle.keys())

    def __iter__(self):
        return iter(sorted(self.names))

    def shape(self, name):
        return tuple(self.handle.get_slice(name).get_shape())

    def dtype(self, name):
        return {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "I64": "int64", "I32": "int32", "BOOL": "bool"}.get(
            self.handle.get_slice(name).get_dtype(), self.handle.get_slice(name).get_dtype())

    def tensor(self, name):
        return self.handle.get_tensor(name)


def tensor_to_mlx(tensor):
    import mlx.core as mx
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.contiguous().view(torch.uint16).numpy()).view(mx.bfloat16)
    return mx.array(tensor.contiguous().numpy())


def component_source(source_root: Path, component: str):
    uf = source_root / "uniforge3d" / "uniforge3d"
    if component in ("shape512", "shape1024", "texture"):
        name = {"shape512": "stage2_shape512", "shape1024": "stage2_shape1024", "texture": "stage3_texture1024"}[component]
        path = uf / component / f"uniforge3d_{name}_denoiser.pt"
        raw = load_torch_state(path)
        state = {k.removeprefix("base_model.model."): v for k, v in raw.items()}
        return TorchSource([state]), [path]
    if component in ("shape512_adapter", "shape1024_adapter", "texture_adapter"):
        stage = component.removesuffix("_adapter")
        name = {"shape512": "stage2_shape512", "shape1024": "stage2_shape1024", "texture": "stage3_texture1024"}[stage]
        path = uf / stage / f"uniforge3d_{name}_vggt_adapter.pt"
        return TorchSource([load_torch_state(path)]), [path]
    if component == "vggt_conditioner":
        path = uf / "vggt" / "uniforge3d_shape_texture_vggt_conditioner.pt"
        return TorchSource([load_torch_state(path)]), [path]
    if component in ("structure_dit", "structure_vggt", "structure_fusion"):
        joint_path = uf / "stage1" / "uniforge3d_stage1_structure_joint_model.pt"
        joint = load_torch_state(joint_path)
        joint = {
            re.sub(r"^fusion_groups\.(\d+)\.f_from_z\.", r"dit_to_vggt.\1.", k): v
            for k, v in joint.items() if not re.match(r"^fusion_groups\.\d+\.z_from_f\.", k)
        }
        if component == "structure_fusion":
            return TorchSource([{k: v for k, v in joint.items() if not k.startswith(("dit.", "vggt."))}]), [joint_path]
        prefix = "dit." if component == "structure_dit" else "vggt."
        overlay = {k[len(prefix):]: v for k, v in joint.items() if k.startswith(prefix)}
        if component == "structure_dit":
            base_path = source_root / "trellis2" / "ckpts" / "ss_flow_img_dit_1_3B_64_bf16.safetensors"
            from safetensors.torch import load_file
            base = load_file(str(base_path), device="cpu")
        else:
            base_path = uf / "vggt" / "uniforge3d_stage1_structure_vggt_backbone.pt"
            base = {k.removeprefix("model."): v for k, v in load_torch_state(base_path).items()}
            base = {k: v for k, v in base.items() if not k.startswith("camera_head.") and k != "aggregator.patch_embed.mask_token"}
        return TorchSource([base, overlay], scale=2.0 if component == "structure_dit" else 1.0), [base_path, joint_path]
    paths = {
        "structure_decoder": source_root / "trellis1/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "shape_decoder": source_root / "trellis2/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "texture_decoder": source_root / "trellis2/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "dinov3": source_root / "dinov3/model.safetensors",
        "rmbg": source_root / "rmbg/model.safetensors",
    }
    path = paths[component]
    return SafeSource(path), [path]


COMPONENTS = (
    "structure_dit", "structure_vggt", "structure_fusion", "vggt_conditioner",
    "shape512", "shape512_adapter", "shape1024", "shape1024_adapter", "texture", "texture_adapter",
    "structure_decoder", "shape_decoder", "texture_decoder", "rmbg", "dinov3",
)
