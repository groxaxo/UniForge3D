# Complete UniForge3D oMLX model bundle

All **15 inference components** are present in
`weights/UniForge3D-oQ5-complete/`. The original fourteen selective oMLX oQ5
components are combined with the recovered public DINOv3 encoder.
This is the recommended quality-focused local bundle.

The bundle has **9.922 GB** of logical weight
storage, versus **32.344 GB** of source tensors
after LoRA merging: **69.3% smaller**.
Its weight payloads use local hard links to the retained checkpoints, saving
duplicate disk space. Every component path is inside the bundle; copying this
directory to another volume produces a self-contained copy. Treat weights as
immutable because modifying a hard-linked file also modifies its retained copy.

## Public DINOv3 source

- Repository: [onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX](https://huggingface.co/onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX).
- Pinned revision: `9f74690bbbba9d76c4cce9062195fc7c95e23bb5`.
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
| Attention + MLP oQ5 | 219.7 | 0.999495540 | 0.994968564 | Below threshold |
| MLP oQ5, FP16 attention | 349.1 | 0.999759356 | 0.997464746 | Pass |
| FP16 projections, FP32 boundaries **selected** | 608.6 | 0.999999351 | 0.999984176 | Pass |

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
- 16 converter/regression tests pass. The complete bundle has per-file
  SHA-256 hashes; its verifier also runs a real multiview encoder smoke test.
- Before inference, the idle Bonsai 27B model was unloaded from LM Studio and
  its loaded-model list was confirmed empty. Runtimes and GPU tests ran
  sequentially, with their caches released between processes.

The evidence on this page covers the **complete model bundle and DINO encoder**.
The subsequent [local MLX runtime](MLX_RUNTIME.md) now exercises the complete
single-view cascade through a textured GLB, with additional component parity,
two-view conditioning, background-removal and rendered-output checks. The
existing CUDA/Studio workflow remains separate. Encoder cosine similarity and
one successful 3D result do not establish equivalence to the complete
unquantized CUDA pipeline.

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
