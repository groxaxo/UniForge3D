#!/usr/bin/env python3
"""Select a tested vision candidate and assemble a complete local model bundle."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_onnx_dinov3 import FOLDERS
from tools.validate_onnx_dinov3 import candidate_gate
from uniforge3d_mlx.onnx_dinov3 import REPO, REVISION, sha256, verify_download


def choose_candidate(candidates, expected_inputs, source_sha256, *, preference="fidelity"):
    accepted = []
    for candidate in candidates:
        parity = candidate["parity"]
        if parity["inputs"] != expected_inputs or parity["source_sha256"] != source_sha256:
            raise ValueError("Candidate was evaluated against different frozen inputs")
        if len(parity["tests"]) != len(expected_inputs):
            raise ValueError("Candidate evaluation is incomplete")
        if [dict((k, t[k]) for k in ("case", "split", "size", "image_sha256"))
            for t in parity["tests"]] != expected_inputs:
            raise ValueError("Candidate test identity mismatch")
        candidate["accepted"] = all(candidate_gate(test) for test in parity["tests"])
        if candidate["accepted"]:
            multiview = parity.get("multiview", {})
            if multiview.get("cosine", 0) < .9999 or multiview.get("minimum_token_cosine", 0) < .999:
                raise ValueError("Passing candidate lacks a passing multiview test")
            accepted.append(candidate)
    if not accepted:
        raise ValueError("No tested candidate meets the encoder fidelity gate")
    if preference == "compact":
        return min(accepted, key=lambda c: c["weight_bytes"])
    if preference != "fidelity":
        raise ValueError("Unknown selection preference")
    return max(accepted, key=lambda c: (
        min(t["cosine"] for t in c["parity"]["tests"]),
        min(t["minimum_token_cosine"] for t in c["parity"]["tests"]),
        -c["weight_bytes"],
    ))


def materialize_component(source, destination):
    """Hard-link immutable weight payloads; copy small, editable metadata."""
    destination.mkdir()
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        target = destination / path.name
        if path.suffix == ".safetensors":
            os.link(path, target)
        else:
            shutil.copy2(path, target)


def documentation(report):
    labels = {"oq5": "Attention + MLP oQ5", "mlp": "MLP oQ5, FP16 attention",
              "fp16": "FP16 projections, FP32 boundaries"}
    rows = []
    for candidate in report["candidates"]:
        tests = candidate["parity"]["tests"]
        mark = " **selected**" if candidate["label"] == report["selected_vision_policy"] else ""
        rows.append(f"| {labels[candidate['label']]}{mark} | {candidate['weight_bytes']/1e6:.1f} | "
                    f"{min(t['cosine'] for t in tests):.9f} | "
                    f"{min(t['minimum_token_cosine'] for t in tests):.9f} | "
                    f"{'Pass' if candidate['accepted'] else 'Below threshold'} |")
    return f"""# Complete UniForge3D oMLX model bundle

All **15 inference components** are present in
`weights/UniForge3D-oQ5-complete/`. The original fourteen selective oMLX oQ5
components are combined with the recovered public DINOv3 encoder.
This is the recommended quality-focused local bundle.

The bundle has **{report['logical_weight_bytes']/1e9:.3f} GB** of logical weight
storage, versus **{report['source_logical_bytes']/1e9:.3f} GB** of source tensors
after LoRA merging: **{report['size_reduction_percent']:.1f}% smaller**.
Its weight payloads use local hard links to the retained checkpoints, saving
duplicate disk space. Every component path is inside the bundle; copying this
directory to another volume produces a self-contained copy. Treat weights as
immutable because modifying a hard-linked file also modifies its retained copy.

## Public DINOv3 source

- Repository: [{REPO}](https://huggingface.co/{REPO}).
- Pinned revision: `{REVISION}`.
- Both the repository metadata and actual weight-file access were checked
  without credentials. The complete FP32 export was downloaded with `token=False`.
- Graph and external-weight file sizes and SHA-256 values match Hugging Face.
- All 415 recovered tensors strictly load into the installed HF architecture.
  All 144 ONNX projection matrices were transposed into HF/MLX layout. Graph
  checks cover block inputs and biases; every external learned tensor is used.
- Query/value biases eliminated from the inference graph are represented as
  zeros. The unused training mask token has an explicit zero placeholder.
  Masked-image inference is outside this conversion's contract.
- Numerical parity is against this public ONNX export. A direct byte comparison
  with the gated Meta checkpoint was not possible. This is not the SoftEq
  segmentation fine-tune.

## Measured precision choice

The original fourteen components retain their selective oMLX oQ5 policy:
1,383 Q5, 14 Q6, and 115 Q8 matrices, with decoders, custom fusion/adapters,
heads, convolutions and other sensitive boundaries preserved.

DINO candidates use activation diagonals collected from **two calibration
images** at both 512px and 1024px. Two different images are held out from
calibration. The same eight image/resolution pairs are evaluated against real
ONNX Runtime inference. ONNX's pre-final-norm output is normalized with
UniForge3D's non-affine layer norm; the public ONNX outputs remain available.
Original-image hashes and frozen reference hashes are recorded.

The acceptance threshold is feature cosine >=0.999 and minimum token cosine
>=0.995 on every case. The default then favors the highest measured fidelity.
The table reports the **worst** result across all eight cases.

| DINO storage policy | MB | Minimum feature cosine | Minimum token cosine | Accuracy gate |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

The selected FP16 candidate retains all 271 non-projection tensors at FP32;
only the 144 internal attention/MLP matrices use FP16 storage. Activations remain
FP32 in this MLX encoder. It costs about 260 MB more than the passing MLP-oQ5
candidate, about 2.6% of the whole bundle, and preserves features much more
closely. The MLP-oQ5 candidate remains available at
`weights/UniForge3D-oQ5-ONNX-MLP/dinov3/` as a measured compact alternative.
The fully quantized attention/MLP candidate missed the minimum-token threshold
on one held-out 1024px case and is not selected.

## Validation

- ONNX Runtime CPU inference ran on all four images at both resolutions.
- Recovered PyTorch MPS features and MLX source features passed numerical
  comparison against ONNX on every case.
- Both quantized candidates passed the packed-tensor audit, including native
  MLX quantized matmul against an independent NumPy FP64 reference.
- Every selected FP16 cast and every preserved FP32 tensor was compared with
  the recovered source. All fourteen existing components were audited again.
- Multiview inputs return `[1,2,1029,1024]` at 512px and match individual-view
  execution. Single-view 512px/1024px outputs are `[1,1029,1024]` and
  `[1,4101,1024]`.
- {report['regression_tests']['passed']} converter/regression tests pass. The complete bundle has per-file
  SHA-256 hashes; its verifier also runs a real multiview encoder smoke test.
- Before inference, the idle Bonsai 27B model was unloaded from LM Studio and
  its loaded-model list was confirmed empty. Runtimes and GPU tests ran
  sequentially, with their caches released between processes.

This report covers the **complete model bundle and DINO encoder**.
The dedicated local MLX runtime is implemented separately; see the
repository-root `MLX_RUNTIME.md` for its current end-to-end evidence, numerical
checks and limitations. Bundle assembly does not itself run the image-to-3D
cascade. The existing CUDA/Studio workflow remains separate. Encoder cosine
similarity does not establish complete 3D-output equivalence.

## Use and reproduce

```python
from uniforge3d_mlx.dinov3 import DinoV3Encoder

encoder = DinoV3Encoder("weights/UniForge3D-oQ5-complete/dinov3")
# Float RGB in [0,1], [B,3,H,W] or [B,V,3,H,W], square and divisible by 16.
features = encoder(images)
```

Verify the completed local bundle and run its actual encoder:

```sh
.venv/bin/python tools/verify_model_bundle.py --smoke
.venv/bin/python -m pytest -q tests
```

For a fresh build after the fourteen baseline components are prepared:

```sh
.venv/bin/python tools/prepare_onnx_dinov3.py
.venv/bin/python tools/validate_onnx_dinov3.py --phase onnx
.venv/bin/python tools/validate_onnx_dinov3.py --phase source
.venv/bin/python tools/build_onnx_dinov3.py --policy oq5
.venv/bin/python tools/validate_onnx_dinov3.py --phase candidate --label oq5 --candidate weights/UniForge3D-oQ5-ONNX/dinov3
.venv/bin/python tools/build_onnx_dinov3.py --policy mlp
.venv/bin/python tools/validate_onnx_dinov3.py --phase candidate --label mlp --candidate weights/UniForge3D-oQ5-ONNX-MLP/dinov3
.venv/bin/python tools/build_onnx_dinov3.py --policy fp16
.venv/bin/python tools/validate_onnx_dinov3.py --phase candidate --label fp16 --candidate weights/UniForge3D-oQ5-ONNX-FP16/dinov3
.venv/bin/python -m pytest -q tests > weights/work/onnx-tests.log
.venv/bin/python tools/report_onnx_dinov3.py
.venv/bin/python tools/verify_model_bundle.py --smoke
```

Builders refuse to overwrite completed artifacts. `bundle.json`,
`verification.json`, and `evidence/` contain the recorded sources, precision
selection, component audits, feature measurements, and regression results.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference", choices=("fidelity", "compact"), default="fidelity")
    args = parser.parse_args()
    from tools.audit_uniforge3d_mlx import audit_component
    from uniforge3d_mlx.source import COMPONENTS
    import numpy as np
    from safetensors import safe_open

    base = ROOT / "weights/UniForge3D-oQ5"
    output = ROOT / ("weights/UniForge3D-oQ5-complete" if args.preference == "fidelity"
                     else "weights/UniForge3D-oQ5-complete-compact")
    work = ROOT / "weights/work/onnx-validation"
    source = ROOT / "weights/source/dinov3-onnx-hf"
    preparation = json.loads((source / "preparation.json").read_text())
    test_log = ROOT / "weights/work/onnx-tests.log"
    test_summary = test_log.read_text().strip().splitlines()[-1]
    test_match = re.fullmatch(r"(\d+) passed in [\d.]+s", test_summary)
    if test_match is None:
        raise ValueError("A successful converter regression run is required")
    reference = json.loads((work / "onnx-reference.json").read_text())
    source_parity = json.loads((work / "source-parity.json").read_text())
    verify_download(ROOT / "weights/source/dinov3-onnx")
    if sha256(source / "model.safetensors") != preparation["normalized_sha256"]:
        raise ValueError("Prepared source changed")
    if not source_parity["all_passed"] or source_parity["inputs"] != reference["inputs"]:
        raise ValueError("Source MLX parity did not pass")
    candidates = []
    for label, folder in FOLDERS.items():
        root = ROOT / "weights" / folder / "dinov3"
        candidate = {"label": label, "path": str(root),
                     "weight_bytes": sum(p.stat().st_size for p in root.glob("*.safetensors")),
                     "parity": json.loads((work / f"{label}-parity.json").read_text())}
        if label == "fp16":
            precision = json.loads((root / "precision-verification.json").read_text())
            if sha256(root / "model.safetensors") != precision["sha256"]:
                raise ValueError("FP16 candidate changed after verification")
            # Independently compare every selected FP16 tensor or preserved FP32
            # boundary against the recovered source, not only against a hash.
            from uniforge3d_mlx.quantization import eligible
            with safe_open(str(source / "model.safetensors"), framework="numpy") as original, \
                 safe_open(str(root / "model.safetensors"), framework="numpy") as written:
                if set(original.keys()) != set(written.keys()):
                    raise ValueError("Selected tensor inventory differs")
                for name in original.keys():
                    value = original.get_tensor(name)
                    expected = value.astype(np.float16) if eligible("dinov3", name, value.shape, str(value.dtype)) else value
                    actual = written.get_tensor(name)
                    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                        raise ValueError(f"Selected tensor differs from intended source conversion: {name}")
            candidate["audit"] = {"status": "passed", **precision}
        else:
            print("AUDIT CANDIDATE", label, flush=True)
            candidate["audit"] = audit_component(root)
        candidates.append(candidate)
    selected = choose_candidate(candidates, reference["inputs"], preparation["normalized_sha256"],
                                preference=args.preference)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite complete bundle: {output}")
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(exist_ok=False)
    components = {}
    for name in COMPONENTS:
        component_source = Path(selected["path"]) if name == "dinov3" else base / name
        print("ASSEMBLE AND AUDIT", name, flush=True)
        materialize_component(component_source, staging / name)
        audit = selected["audit"] if name == "dinov3" else audit_component(staging / name)
        components[name] = {
            "path": name,
            "format": "huggingface-safetensors" if name == "dinov3" and selected["label"] == "fp16"
                      else "uniforge3d-mlx-omlx-oq5-v1",
            "audit": audit,
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in sorted((staging / name).iterdir()) if p.is_file()},
        }
    if len(components) != 15 or not all(c["audit"]["status"] == "passed" for c in components.values()):
        raise ValueError("Incomplete model bundle")
    if (ROOT / "weights/source/dinov3/LICENSE.md").is_file():
        shutil.copy2(ROOT / "weights/source/dinov3/LICENSE.md", staging / "dinov3/LICENSE.md")
        path = staging / "dinov3/LICENSE.md"
        components["dinov3"]["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    original = json.loads((base / "build-report.json").read_text())["source_logical_bytes_after_lora_merge"]
    with safe_open(str(source / "model.safetensors"), framework="numpy") as handle:
        original += sum(int(np.prod(handle.get_slice(n).get_shape())) * 4 for n in handle.keys())
    total = sum(f["bytes"] for c in components.values() for n, f in c["files"].items() if n.endswith(".safetensors"))
    report = {
        "format": "uniforge3d-complete-model-bundle-v1", "component_count": 15,
        "all_required_inference_weights_present": True, "all_component_integrity_audits_passed": True,
        "dinov3_source": {"repo": REPO, "revision": REVISION, "anonymous_download_verified": True,
                         "recovered_source_sha256": preparation["normalized_sha256"]},
        "selected_vision_policy": selected["label"],
        "selection_preference": args.preference,
        "selection_rule": ("highest minimum feature cosine among passing candidates" if args.preference == "fidelity"
                           else "smallest passing candidate"),
        "acceptance_gate": "cosine >=0.999 and minimum token cosine >=0.995 on every frozen image/resolution, plus multiview consistency",
        "logical_weight_bytes": total, "source_logical_bytes": original,
        "size_reduction_percent": 100 * (1 - total / original),
        "payload_storage": "local hard links to immutable checkpoints; all component paths are inside this bundle",
        "components": components, "candidates": candidates,
        "source_precision_parity": source_parity,
        "full_image_to_3d_mlx_runtime_implemented": True,
        "runtime_entrypoint": "tools/run_mlx.py",
        "runtime_validation_evidence": "MLX_RUNTIME.md; recorded separately from immutable model weights",
        "full_image_to_3d_quality_verified": False,
        "validation_scope": "all component artifact integrity; DINO features on four images at 512/1024; 2 calibration and 2 held-out; multiview",
        "environment": {p: importlib.metadata.version(p) for p in
                        ("onnx", "onnxruntime", "torch", "mlx", "transformers")},
        "regression_tests": {"passed": int(test_match[1]), "summary": test_summary,
                             "log_sha256": sha256(test_log)},
    }
    (staging / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    evidence = staging / "evidence"
    evidence.mkdir()
    for name in ("onnx-reference.json", "torch-parity.json", "source-parity.json",
                 "oq5-parity.json", "mlp-parity.json", "fp16-parity.json", "calibration.json"):
        shutil.copy2(work / name, evidence / name)
    shutil.copy2(ROOT / "weights/source/dinov3-onnx/download-verification.json", evidence / "download-verification.json")
    memory_preflight = ROOT / "weights/work/onnx-memory-preflight.json"
    if memory_preflight.is_file():
        shutil.copy2(memory_preflight, evidence / "memory-preflight.json")
    shutil.copy2(test_log, evidence / "tests.log")
    if args.preference == "fidelity":
        text = documentation(report)
        (staging / "README.md").write_text(text)
        (ROOT / "ONNX_DINOV3.md").write_text(text)
    os.replace(staging, output)
    print(f"COMPLETE BUNDLE: {output}; {total/1e9:.3f} GB; selected={selected['label']}", flush=True)


if __name__ == "__main__":
    main()
