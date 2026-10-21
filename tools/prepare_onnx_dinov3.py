#!/usr/bin/env python3
"""Download without credentials and recover the pinned FP32 DINOv3 export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    import onnx
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file, load_file
    from uniforge3d_mlx.onnx_dinov3 import (
        REPO, REVISION, FILES, verify_download, extract_state, sha256, add_reference_output,
    )
    from uniforge3d_mlx.softeq import torch_backbone

    source = ROOT / "weights/source/dinov3-onnx"
    output = ROOT / "weights/source/dinov3-onnx-hf"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared source: {output}")
    if not args.offline:
        snapshot_download(REPO, revision=REVISION, local_dir=source, token=False,
                          allow_patterns=[*FILES, "README.md", "config.json", "preprocessor_config.json"],
                          max_workers=2)
    verification = {"repo": REPO, "revision": REVISION, "download_token": False,
                    "files": verify_download(source)}
    (source / "download-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    graph = source / "onnx/model.onnx"
    onnx.checker.check_model(str(graph))
    state, mapping = extract_state(graph)
    config = json.loads((source / "config.json").read_text())
    if (config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"],
        config["intermediate_size"], config["num_register_tokens"], config["patch_size"],
        config["model_type"], config["use_gated_mlp"]) != (1024, 24, 16, 4096, 4, 16, "dinov3_vit", False):
        raise ValueError("Unexpected DINOv3 architecture")
    torch.set_num_threads(4)
    model = torch_backbone(config, {k: torch.from_numpy(v.copy()) for k, v in state.items()})
    assert len(model.state_dict()) == 415
    del model
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    save_file(state, str(staging / "model.safetensors"), metadata={"format": "pt"})
    written = load_file(str(staging / "model.safetensors"))
    for name, expected in state.items():
        if written[name].dtype != expected.dtype or not (written[name] == expected).all():
            raise ValueError(f"Recovery readback failed: {name}")
    for name in ("config.json", "preprocessor_config.json", "README.md"):
        shutil.copy2(source / name, staging / name)
    raw_output = add_reference_output(graph, graph.with_name("model_reference.onnx"))
    report = {
        "repo": REPO, "revision": REVISION, "source_verification": verification,
        "source_precision": "float32", "tensor_count": len(state),
        "all_recovered_tensors_verified": True, "strict_hf_load": True,
        "mapping": mapping, "normalized_sha256": sha256(staging / "model.safetensors"),
        "reference_graph": str(graph.with_name("model_reference.onnx")),
        "reference_graph_sha256": sha256(graph.with_name("model_reference.onnx")),
        "reference_raw_output": raw_output,
        "equivalence_scope": "public ONNX inference graph; direct gated Meta checkpoint comparison unavailable",
        "omitted_training_parameter": "mask_token set to zero; no masked-image inference supported",
    }
    (staging / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    os.replace(staging, output)
    print(json.dumps({k: v for k, v in report.items() if k not in ("mapping", "source_verification")}, indent=2))


if __name__ == "__main__":
    main()
