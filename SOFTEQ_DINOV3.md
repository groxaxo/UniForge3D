# User-selected SoftEq DINOv3 replacement

The requested public model is prepared and usable as a local DINOv3 feature
encoder. The selected bundle is `weights/UniForge3D-oQ5-SoftEq-FP16/` and its
machine-readable component map is `bundle.json`.

It reuses the fourteen existing selective oMLX oQ5 components by relative path
and adds this model's **FP16 vision backbone**. This gives fifteen selected
components with **9.919 GB** of logical weight storage. The
existing fourteen component checkpoints were not changed or duplicated.

## Source and adaptation

- Repository: https://huggingface.co/ashiq24/softeq-dinov3-vitl16-pretrain-lvd1689m-ade-seg-c360-s0.1-sp0.8
- Pinned revision: `f853065890a0df7869f9b5267361cdc0ac139fee`.
- The 1.222 GB source checkpoint matches publisher SHA-256
  `c2c6df12b078eecbf00cea076fabdae803dafa01bf53f26648ebdcb01eb0f905`.
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
`53b5af0b1fb1b3b085bde938bb0fb9a63722e3dba12a511925cb247b8cffca84`. Cosine similarity is an encoder feature metric,
not a 3D-output quality score.

| Vision precision | Weight MB | Feature cosine, 512px | Feature cosine, 1024px |
|---|---:|---:|---:|
| Attention + MLP oQ5 | 219.7 | 0.989205 | 0.982831 |
| MLP-only oQ5 | 550.4 | 0.994941 | 0.988610 |
| FP16 (selected) | 606.3 | 0.999971 | 0.999959 |

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
