# UniForge3D selective oMLX oQ5 conversion

**Current recommended bundle:** all fifteen inference components are available
in `weights/UniForge3D-oQ5-complete/`, using the public DINOv3 ONNX export.
See [ONNX_DINOV3.md](ONNX_DINOV3.md) for source recovery, accuracy comparisons,
the selected FP16 vision policy, and complete-bundle verification. Gated access
is no longer needed for this bundle. The fourteen-component baseline recorded
below is retained as the original conversion history.

The user-selected public SoftEq replacement is now prepared and validated as an
encoder at 512px and 1024px. See [SOFTEQ_DINOV3.md](SOFTEQ_DINOV3.md) for the
selected mixed-precision bundle and its limits. The baseline conversion below
still records the original Meta checkpoint as gated; its fourteen converted
components are retained unchanged.

14 of fifteen model components have been converted. DINOv3 remains blocked
by the publisher's Hugging Face access gate. Both stored Hugging Face credentials
were checked and lacked access; no account settings or terms were changed.

The converted checkpoints occupy **9.313 GB**, versus
**31.132 GB** of logical source tensors after merging the trained LoRA
updates (**70.1% smaller**). Source downloads are
retained separately under `weights/source/`.

## What is ready

- 824 LoRA A/B pairs merged into their correct dense,
  convolution, or transposed-convolution base weights.
- 1383 matrices stored at 5-bit, 14 at 6-bit,
  and 115 at 8-bit, using affine groups of 64.
- 5487 tensors retain their source precision.
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
commit `70f44cfd652d2013dc0f9776dca7518e32101b00`. It ranks eligible matrices by relative
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

The custom loader and packed linear operations in `uniforge3d_mlx` now feed
the [dedicated local MLX runtime](MLX_RUNTIME.md). Its complete single-view
image-to-textured-GLB path has been exercised on an Apple M5, with numerical
reference checks and a rendered output. The existing `infer.py` still uses CUDA. These
checkpoints cannot be served by the stock oMLX LLM server or loaded through
`mlx_lm.generate`.

The reusable TRELLIS MLX code is pinned at
`appautomaton/mlx-spatial@d2cc98ef7fb2166dc7e3b826b487e80f6f5027dc` under
`.deps/mlx-spatial`. The port implements UniForge3D's joint VGGT fusion, routed
attention, geometry conditioning and cascade integration. A paired complete
source-precision/CUDA comparison remains outstanding; a successful rendered
run does not establish the optimal quantization allocation or CUDA equivalence.

## Component inventory

Sizes use decimal GB. Source columns describe effective tensors after LoRA
merging and the original loader's removal of unused Stage1 state.

| Component | Source GB | Output GB | Q5 matrices | Q6 matrices | Q8 matrices | Preserved tensors |
|---|---:|---:|---:|---:|---:|---:|
| rmbg | 0.885 | 0.265 | 84 | 2 | 10 | 658 |
| shape1024 | 5.169 | 0.981 | 185 | 1 | 24 | 430 |
| shape1024_adapter | 0.256 | 0.256 | 0 | 0 | 0 | 32 |
| shape512 | 5.169 | 0.981 | 185 | 1 | 24 | 430 |
| shape512_adapter | 0.252 | 0.252 | 0 | 0 | 0 | 28 |
| shape_decoder | 0.948 | 0.948 | 0 | 0 | 0 | 292 |
| structure_decoder | 0.148 | 0.148 | 0 | 0 | 0 | 74 |
| structure_dit | 3.622 | 0.944 | 200 | 1 | 9 | 430 |
| structure_fusion | 0.644 | 0.644 | 0 | 0 | 0 | 296 |
| structure_vggt | 3.767 | 0.789 | 269 | 4 | 15 | 983 |
| texture | 5.169 | 0.981 | 191 | 1 | 18 | 430 |
| texture_adapter | 0.256 | 0.256 | 0 | 0 | 0 | 32 |
| texture_decoder | 0.948 | 0.948 | 0 | 0 | 0 | 284 |
| vggt_conditioner | 3.898 | 0.920 | 269 | 4 | 15 | 1088 |

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
