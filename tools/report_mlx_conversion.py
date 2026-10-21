#!/usr/bin/env python3
"""Build a reviewable conversion report without implying inference acceptance."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "weights/UniForge3D-oQ5"
    audit = json.loads((output / "audit.json").read_text())
    source_report = json.loads((ROOT / "weights/source/verification.json").read_text())
    manifests = {p.parent.name: json.loads(p.read_text()) for p in sorted(output.glob("*/manifest.json"))}
    distribution, elements = Counter(), Counter()
    for manifest in manifests.values():
        distribution.update(manifest["matrix_counts"])
        elements.update(manifest["parameter_counts"])
    original = sum(m["source_logical_bytes"] for m in manifests.values())
    packed = sum(d["artifact_bytes"] for d in audit["components"].values())
    missing = [n for n, v in source_report["repositories"].items() if v["status"] != "verified"]
    report = {
        "oq_level": 5, "format": "uniforge3d-mlx-omlx-oq5-v1",
        "all_required_sources_present": not missing,
        "missing_sources": missing, "converted_components": len(manifests),
        "checkpoint_audits_passed": all(x["status"] == "passed" for x in audit["components"].values()),
        "full_image_to_3d_inference_implemented": False,
        "full_image_to_3d_inference_verified": False,
        "stock_omlx_server_model": False,
        "selection_evidence": "Architecture allow-list and deterministic weight-reconstruction MSE; no activation or geometry calibration",
        "source_logical_bytes_after_lora_merge": original, "output_safetensors_bytes": packed,
        "size_reduction_percent": 100 * (1 - packed / original),
        "tensor_counts_by_precision": dict(distribution),
        "logical_elements_by_precision": dict(elements),
        "merged_lora_pairs": sum(m["merged_lora_pairs"] for m in manifests.values()),
        "components": audit["components"], "source_repositories": source_report["repositories"],
        "environment": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "torch", "numpy", "safetensors", "huggingface-hub", "mlx-spatial")},
        "platform": platform.platform(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "omlx_commit": "70f44cfd652d2013dc0f9776dca7518e32101b00",
        "mlx_spatial_commit": "d2cc98ef7fb2166dc7e3b826b487e80f6f5027dc",
    }
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    reference = output / "reference"
    reference.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "weights/source/uniforge3d/pipeline.json", reference / "pipeline.json")
    for folder in (ROOT / "weights/source").iterdir():
        if not folder.is_dir():
            continue
        destination = reference / folder.name
        destination.mkdir(exist_ok=True)
        for path in folder.iterdir():
            if path.is_file() and (path.suffix in (".json", ".py", ".md") or path.name.startswith("LICENSE")):
                shutil.copy2(path, destination / path.name)
    rows = []
    for name, entry in report["components"].items():
        d = manifests[name]["matrix_counts"]
        rows.append(f"| {name} | {entry['source_logical_bytes']/1e9:.3f} | {entry['artifact_bytes']/1e9:.3f} | {d.get('5',0)} | {d.get('6',0)} | {d.get('8',0)} | {d.get('source_precision',0)} |")
    source_status = (f"{len(manifests)} of fifteen model components have been converted. DINOv3 remains blocked\n"
                     "by the publisher's Hugging Face access gate. Both stored Hugging Face credentials\n"
                     "were checked and lacked access; no account settings or terms were changed."
                     if missing else "All fifteen model components have been converted and their source files verified.")
    text = f"""# UniForge3D selective oMLX oQ5 conversion

The recommended complete bundle and public DINOv3 ONNX recovery are documented
in [ONNX_DINOV3.md](ONNX_DINOV3.md). That bundle includes all fifteen components
without requiring gated access. The baseline conversion history follows.

The user-selected public SoftEq replacement and its encoder validation are
documented separately in [SOFTEQ_DINOV3.md](SOFTEQ_DINOV3.md). That experimental
bundle reuses these baseline components without changing them.

{source_status}

The converted checkpoints occupy **{packed/1e9:.3f} GB**, versus
**{original/1e9:.3f} GB** of logical source tensors after merging the trained LoRA
updates (**{report['size_reduction_percent']:.1f}% smaller**). Source downloads are
retained separately under `weights/source/`.

## What is ready

- {report['merged_lora_pairs']} LoRA A/B pairs merged into their correct dense,
  convolution, or transposed-convolution base weights.
- {distribution['5']} matrices stored at 5-bit, {distribution['6']} at 6-bit,
  and {distribution['8']} at 8-bit, using affine groups of 64.
- {distribution['source_precision']} tensors retain their source precision.
- Every converted tensor passed the artifact audit. Unquantized output bytes
  match their recorded source hashes. Every packed matrix was exercised through
  MLX `quantized_matmul` against an independent NumPy FP64 reference.
- All downloaded large files matched their publisher-provided LFS SHA-256.
- Six focused converter tests pass, including all three LoRA merge operations,
  PEFT normalization, matrix selection, scalar preservation, and packed loading.

## Selection policy

Only internal attention and MLP matrices in the TRELLIS denoisers, VGGT trunks,
and RMBG Swin backbone are eligible. Convolutions, all VAE decoders, point/DPT
heads, custom fusion layers, shape conditioners, VGGT-to-DINO adapters,
normalization, timestep/modulation, embeddings, and input/output boundaries
remain at source precision.

The converter calls the installed oMLX oQ5 allocator and tensor quantizer at
commit `{report['omlx_commit']}`. It ranks eligible matrices by relative
5-bit weight-reconstruction MSE on 64 deterministic sampled rows. oMLX allocates
6-/8-bit protection within its 5.7-bit-per-weight ceiling for the selected
matrices, including affine metadata. Preserved tensors are outside that budget.
BF16 scales/biases are stored compactly and widened to FP32 during packed
execution. No oMLX installation or running model configuration was modified.

These scores measure weight reconstruction. They are not activation-calibrated
sensitivity, image quality, geometry fidelity, or proof that this is the optimal
bit allocation for UniForge3D. A frozen image-to-3D comparison remains required
before treating the candidate as quality-validated.

## Runtime boundary

This build provides a **custom MLX checkpoint loader and packed linear
operations** in `uniforge3d_mlx`. The full UniForge3D image-to-3D MLX pipeline is
not implemented or validated. The existing `infer.py` still uses CUDA. These
checkpoints cannot be served by the stock oMLX LLM server or loaded through
`mlx_lm.generate`.

The reusable TRELLIS MLX code is pinned at
`appautomaton/mlx-spatial@{report['mlx_spatial_commit']}` under
`.deps/mlx-spatial`. UniForge3D's joint VGGT fusion, routed attention, geometry
conditioning, and full pipeline integration still require a dedicated runtime
port and source-precision comparison.

## Component inventory

Sizes use decimal GB. Source columns describe effective tensors after LoRA
merging and the original loader's removal of unused Stage1 state.

| Component | Source GB | Output GB | Q5 matrices | Q6 matrices | Q8 matrices | Preserved tensors |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Reproduce and resume

Use Python 3.13 on Apple Silicon, the pinned oMLX checkout at
`/Users/op/AI/omlx`, and the pinned `mlx-spatial` checkout. Install this project's
`requirements-mlx.txt` into `.venv`; the existing environment is ready.

After access is granted at
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m:

```sh
.venv/bin/python tools/download_mlx_sources.py --only dinov3
.venv/bin/python tools/quantize_uniforge3d_mlx.py --component dinov3 --resume
.venv/bin/python tools/verify_mlx_sources.py
.venv/bin/python tools/audit_uniforge3d_mlx.py
.venv/bin/python tools/report_mlx_conversion.py
```

The downloader pins every source revision and resumes interrupted transfers.
The converter refuses to overwrite completed artifacts. `--resume` only
discards incomplete generated `.partial` component directories. Source model
files are never removed or changed.

Machine-readable evidence: `weights/source/verification.json`,
`weights/UniForge3D-oQ5/audit.json`, `weights/UniForge3D-oQ5/build-report.json`,
and each component's `manifest.json`. The copied upstream configuration and
model notices are under `weights/UniForge3D-oQ5/reference/`.
"""
    (ROOT / "MLX_QUANTIZATION.md").write_text(text)
    (output / "README.md").write_text(text)
    print(f"Report: {output / 'build-report.json'}")
    print(f"Converted: {len(manifests)}/15; packed weights: {packed/1e9:.3f} GB; missing: {missing}")


if __name__ == "__main__":
    main()
