#!/usr/bin/env python3
"""Prepare and verify the user-selected, modified DINOv3 feature backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "weights/source/dinov3-softeq")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/source/dinov3-softeq-hf")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    from uniforge3d_mlx.softeq import (BACKBONE_CONFIG, SOFTEQ_REPO, SOFTEQ_REVISION, SOFTEQ_SHA256,
                                      normalize_softeq_state, torch_backbone)
    torch.set_num_threads(4)
    path = args.source / "model.safetensors"
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if digest != SOFTEQ_SHA256:
        raise ValueError("Source checkpoint does not match the pinned publisher SHA-256")
    config = json.loads((args.source / "config.json").read_text())
    backbone_config = json.loads((args.source / BACKBONE_CONFIG).read_text())
    if [backbone_config.get(k) for k in ("hidden_size", "num_hidden_layers", "patch_size", "num_register_tokens")] != [1024, 24, 16, 4]:
        raise ValueError("Backbone dimensions do not match UniForge3D's DINOv3 contract")
    raw = load_file(str(path))
    normalized = normalize_softeq_state(raw, config)
    model = torch_backbone(backbone_config, normalized)
    reference = torch_backbone(backbone_config, normalized, filtered_reference_state=raw, soft_config=config)
    tests = []
    with torch.inference_mode():
        for size in (512, 1024):
            generator = torch.Generator().manual_seed(42)
            pixels = torch.randn((1, 3, size, size), generator=generator)
            expected = reference.embeddings.patch_embeddings(pixels)
            actual = model.embeddings.patch_embeddings(pixels)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            tests.append({"image_size": size, "patch_shape": list(actual.shape),
                          "max_absolute_error": float((actual - expected).abs().max())})
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "model.safetensors").exists():
        raise FileExistsError("Refusing to replace an existing normalized backbone")
    save_file(normalized, str(args.output / "model.safetensors"))
    backbone_config["_name_or_path"] = SOFTEQ_REPO
    backbone_config["uniforge3d_experimental_replacement"] = True
    (args.output / "config.json").write_text(json.dumps(backbone_config, indent=2) + "\n")
    report = {"repo": SOFTEQ_REPO, "revision": SOFTEQ_REVISION, "source_sha256": digest,
              "normalized_sha256": hashlib.file_digest((args.output / "model.safetensors").open("rb"), "sha256").hexdigest(),
              "normalized_tensor_count": len(normalized), "hf_strict_loading": "passed",
              "patch_filter_folding": tests, "positional_filter": "identity by published softness==1.0 shortcut",
              "patch_filter_softness": config["soft_thresholding"], "segmentation_head": "omitted",
              "original_meta_checkpoint_equivalence": "not claimed; this is an ADE20K-finetuned replacement",
              "runtime_code_from_hub_executed": False}
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
