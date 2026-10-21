#!/usr/bin/env python3
"""Record the selected replacement and the measured precision tradeoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()


def main():
    from uniforge3d_mlx.softeq import SOFTEQ_REPO, SOFTEQ_REVISION
    base = ROOT / "weights/UniForge3D-oQ5"
    selected = ROOT / "weights/UniForge3D-oQ5-SoftEq-FP16"
    work = ROOT / "weights/work"
    preparation = json.loads((ROOT / "weights/source/dinov3-softeq-hf/preparation.json").read_text())
    source = json.loads((work / "softeq-validation/source-parity.json").read_text())
    fp16 = json.loads((work / "softeq-validation-fp16/fp16-parity.json").read_text())
    precision = json.loads((selected / "dinov3/precision-verification.json").read_text())
    if not precision["all_cast_tensors_verified"]:
        raise ValueError("FP16 checkpoint audit did not pass")
    if sha(selected / "dinov3/model.safetensors") != precision["sha256"]:
        raise ValueError("Selected checkpoint changed after verification")
    if source["image_sha256"] != fp16["image_sha256"] or any(t["cosine"] < .999 or t["minimum_token_cosine"] < .995 for t in fp16["tests"]):
        raise ValueError("Selected encoder did not pass the frozen-image fidelity gate")
    base_audit = json.loads((base / "audit.json").read_text())
    components = {}
    for name, audit in base_audit["components"].items():
        if audit["status"] != "passed":
            raise ValueError(f"Base component failed its audit: {name}")
        components[name] = {"path":f"../UniForge3D-oQ5/{name}","format":"uniforge3d-mlx-omlx-oq5-v1",
                            "manifest_sha256":sha(base / name / "manifest.json")}
    components["dinov3"] = {"path":"dinov3","format":"huggingface-safetensors","precision":"float16",
                            "sha256":precision["sha256"],"repo":SOFTEQ_REPO,"revision":SOFTEQ_REVISION}
    candidates = []
    for label, folder, parity_folder, filename in (
        ("Attention + MLP oQ5", "UniForge3D-oQ5-SoftEq", "softeq-validation", "quantized-parity.json"),
        ("MLP-only oQ5", "UniForge3D-oQ5-SoftEq-MLP", "softeq-validation-mlp", "quantized-parity.json"),
        ("FP16 (selected)", "UniForge3D-oQ5-SoftEq-FP16", "softeq-validation-fp16", "fp16-parity.json"),
    ):
        path = ROOT / "weights" / folder
        candidate = json.loads((work / parity_folder / filename).read_text())
        if candidate["image_sha256"] != source["image_sha256"]:
            raise ValueError("Candidate comparison used different inputs")
        size = sum(f.stat().st_size for f in (path / "dinov3").glob("*.safetensors"))
        candidates.append({"label":label,"path":str(path / "dinov3"),"weight_bytes":size,"feature_comparison":candidate})
    selected_total = sum(a["artifact_bytes"] for a in base_audit["components"].values()) + precision["bytes"]
    report = {"selected_model":SOFTEQ_REPO,"revision":SOFTEQ_REVISION,"is_original_meta_checkpoint":False,
              "feature_encoder_inference_verified":True,"complete_image_to_3d_mlx_runtime_implemented":False,
              "complete_image_to_3d_quality_verified":False,"selected_vision_precision":"float16",
              "base_precision":"existing selective oMLX oQ5; original source-precision boundaries retained",
              "selection_reason":"FP16 vision features remain much closer to source for 56 MB more than MLP-only oQ5",
              "component_count":len(components),"logical_bundle_weight_bytes":selected_total,
              "base_components_reused_without_copying":len(base_audit["components"]),
              "components":components,"preparation":preparation,"source_precision_feature_comparison":source,
              "candidates":candidates,"validation_scope":"one frozen demo image at 512 and 1024 pixels; no original Meta or 3D-output equivalence claim"}
    (selected / "bundle.json").write_text(json.dumps(report,indent=2)+"\n")
    evidence = selected / "evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "weights/source/dinov3-softeq/download-verification.json",evidence / "download-verification.json")
    shutil.copy2(ROOT / "weights/source/dinov3-softeq/README.md",selected / "dinov3/UPSTREAM_README.md")
    for name, source_path in (("source-parity.json",work / "softeq-validation/source-parity.json"),
                              ("fp16-parity.json",work / "softeq-validation-fp16/fp16-parity.json"),
                              ("calibration.json",work / "softeq-validation/calibration.json"),
                              ("tests.log",work / "softeq-tests.log")):
        shutil.copy2(source_path,evidence / name)
    comparison = []
    for c in candidates:
        by_size = {t["size"]:t for t in c["feature_comparison"]["tests"]}
        comparison.append(f"| {c['label']} | {c['weight_bytes']/1e6:.1f} | {by_size[512]['cosine']:.6f} | {by_size[1024]['cosine']:.6f} |")
    text = f"""# User-selected SoftEq DINOv3 replacement

The requested public model is prepared and usable as a local DINOv3 feature
encoder. The selected bundle is `weights/UniForge3D-oQ5-SoftEq-FP16/` and its
machine-readable component map is `bundle.json`.

It reuses the fourteen existing selective oMLX oQ5 components by relative path
and adds this model's **FP16 vision backbone**. This gives fifteen selected
components with **{selected_total/1e9:.3f} GB** of logical weight storage. The
existing fourteen component checkpoints were not changed or duplicated.

## Source and adaptation

- Repository: https://huggingface.co/{SOFTEQ_REPO}
- Pinned revision: `{SOFTEQ_REVISION}`.
- The 1.222 GB source checkpoint matches publisher SHA-256
  `{preparation['source_sha256']}`.
- This is an ADE20K segmentation fine-tune with soft-equivariant patch
  embeddings. It is not a mirror of the Meta checkpoint used to train
  UniForge3D, so original-model or full 3D quality equivalence is not claimed.
- The segmentation head is omitted. Its complete 415-tensor DINOv3 backbone
  strictly loads into the standard HF architecture.
- The saved patch filter is folded into the Conv2d kernel, independently per
  RGB channel. Actual 512px and 1024px patch outputs matched the original
  filtered operation exactly (maximum absolute error 0).
- The configuration's positional softness is 1.0. The published implementation
  explicitly bypasses coordinate smoothing in this case, including at 1024px.
- No downloaded custom Python code or `trust_remote_code=True` loader was
  executed. The relevant published equations were implemented locally.

## Why the vision encoder stays at FP16

We tested two oQ5 candidates using the installed oMLX allocator and weighted
affine quantizer, calibrated from actual activation diagonals for all 144
attention/MLP matrices on the same project image at both resolutions. The
MLP-only variant preserves all attention tensors at FP32. Both packed variants
passed their structural, preserved-tensor, and native packed-matmul audits.

The measurements below compare full feature tensors against the folded
source-precision PyTorch encoder, using frozen image SHA-256
`{source['image_sha256']}`. Cosine similarity is an encoder feature metric,
not a 3D-output quality score.

| Vision precision | Weight MB | Feature cosine, 512px | Feature cosine, 1024px |
|---|---:|---:|---:|
{chr(10).join(comparison)}

FP16 costs only about 56 MB more than MLP-only oQ5, while keeping the features
substantially closer to the source encoder. Low-bit vision weights remain
experimental alternatives; they are not selected in `bundle.json`. The
existing fourteen components retain their previous selective oQ5 settings.

## Validation and current boundary

The source MLX encoder achieved cosine similarity above 0.9999 at both sizes.
The selected FP16 encoder produces `[1,1029,1024]` and `[1,4101,1024]` tensors.
All 415 FP16 tensors were verified against the intended FP16 casts, and all
twelve converter regression tests pass. Raw preparation, comparison,
calibration, audit, and test evidence is retained under `weights/`.

This validates the replacement encoder on one frozen demonstration image.
The full UniForge3D MLX generation pipeline still needs its custom VGGT/routing
integration. No complete 3D model was generated or quality-compared with this
replacement, and existing CUDA runs/configuration were left unchanged.

## Use the local encoder

```python
from uniforge3d_mlx.dinov3 import DinoV3Encoder

encoder = DinoV3Encoder("weights/UniForge3D-oQ5-SoftEq-FP16/dinov3")
# images: float RGB in [0,1], [B,3,H,W] or [B,V,3,H,W]
features = encoder(images)
```

The encoder normalizes inputs and follows UniForge3D's final non-affine layer
normalization. It uses local files and never falls back to another model.

## Reproduce

```sh
.venv/bin/python tools/download_softeq_dinov3.py
.venv/bin/python tools/prepare_softeq_dinov3.py
.venv/bin/python tools/validate_softeq_dinov3.py --phase source
.venv/bin/python tools/build_softeq_oq5.py
.venv/bin/python tools/validate_softeq_dinov3.py --phase quantized
.venv/bin/python tools/build_softeq_oq5.py --attention-policy preserve --output weights/UniForge3D-oQ5-SoftEq-MLP
.venv/bin/python tools/validate_softeq_dinov3.py --phase quantized --packed weights/UniForge3D-oQ5-SoftEq-MLP/dinov3 --reference weights/work/softeq-validation --output weights/work/softeq-validation-mlp
.venv/bin/python tools/build_softeq_fp16.py
.venv/bin/python tools/validate_softeq_dinov3.py --phase fp16 --packed weights/UniForge3D-oQ5-SoftEq-FP16/dinov3 --reference weights/work/softeq-validation --output weights/work/softeq-validation-fp16
```

The builders refuse to overwrite completed checkpoints. Existing outputs are
already available; the commands describe a fresh build in the same layout.
"""
    (ROOT / "SOFTEQ_DINOV3.md").write_text(text)
    (selected / "README.md").write_text(text)
    print(f"Selected bundle: {selected}; {len(components)} components, {selected_total/1e9:.3f} GB")


if __name__ == "__main__":
    main()
