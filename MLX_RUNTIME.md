# Local UniForge3D MLX runtime

The runtime uses the completed `weights/UniForge3D-oQ5-complete` model bundle.
Neural inference runs in MLX on Apple Silicon. Discrete view routing, mesh
voxelization and parts of remeshing/UV/texture export run on the CPU through
NumPy and the pinned native `mlx-spatial` backend.

The full single-view and four-view cascades have been run and visually inspected on an
Apple M5 with 24 GiB unified memory. The validated example is
`run_logs/mlx-case02-seed42/model.glb`, generated from `assets/case02/001.png`
with the trained default schedule and seed 42. Its `run.json` reports
`status: complete`; a checkpointed run is not a completed GLB.

## Run

The existing `.venv` has the dependencies in `requirements-mlx.txt`. The
editable `mlx-spatial` checkout must remain pinned to
`d2cc98ef7fb2166dc7e3b826b487e80f6f5027dc`.

Unload other GPU models before inference, then run:

```sh
.venv/bin/python tools/run_mlx.py \
  assets/case02/001.png \
  --output run_logs/my-mlx-run
```

Pass multiple image paths for multiple views; the first view is the anchor.
The output directory must be new. The trained default schedule uses 12
structure steps and 20 steps each for Shape512, Shape1024 and texture.
`--steps N` overrides those defaults for diagnostics.

Each completed stage saves an NPZ checkpoint. Resume only with the same
images, bundle, configuration and seed:

```sh
.venv/bin/python tools/run_mlx.py \
  assets/case02/001.png \
  --output run_logs/my-mlx-run --resume
```

`--stop-after structure|shape512|handoff|shape1024|texture|decode` supports
stage-level debugging. A run writes `model.glb` only after all stages and
texture export succeed. Input crops, intermediate fields, subdivisions,
timings, memory high-water mark and artifact checksums remain in the run
directory.

## Completed example

[Four-view render](run_logs/mlx-case02-seed42/model.preview.png) ·
[GLB](run_logs/mlx-case02-seed42/model.glb) ·
[Run record](run_logs/mlx-case02-seed42/run.json) ·
[Export diagnostics](run_logs/mlx-case02-seed42/diagnostics.json) ·
[Combined validation](run_logs/mlx-case02-seed42/validation.json)

- GLB size: 55,591,860 bytes, with 632,644 serialized vertices and 995,066
  triangles across ten primitives (chunked for 16-bit index compatibility).
- Embedded 2048 × 2048 base-color/alpha and metallic/roughness textures.
- Independent Trimesh inspection found finite vertices and valid indices.
  Three.js loaded and rendered all ten textured primitives in four views.
- All thirteen saved stage checkpoint hashes and the final GLB hash were
  verified. Normals and UVs are finite; UVs are within the unit square.
  Both embedded PBR maps have the expected dimensions.
- Native topology diagnostics found no nonmanifold edges or open boundary
  chains/loops after export processing.
- Visual review found a coherent figure, with the reference pose, base,
  hair ornaments, dangling hand accessory and clothing patterns preserved.
  Unseen surfaces were generated from a single image.

The recorded stage times sum to about **29.2 minutes**. This was a staged,
resumed validation run; that sum excludes validation pauses and an interrupted
FP32 high-resolution trial. It is not an uninterrupted wall-clock benchmark.

| Stage | Recorded seconds |
| --- | ---: |
| Preprocessing and DINOv3 | 4.2 |
| Structure generation | 281.5 |
| Shape512 generation | 180.2 |
| 512 mesh and 64-grid handoff | 79.6 |
| Shape1024 generation | 379.4 |
| Texture generation | 350.1 |
| Shape and texture decoding | 137.3 |
| Mesh, UV, texture bake and export | 339.9 |

UV construction accounted for 291.0 seconds of export time; texture baking
took 11.8 seconds. The recorded MLX allocation peak was 10.05 GB; this excludes
CPU allocations and unrelated applications. Other loaded GPU models were
unloaded before testing, and heavy jobs ran sequentially.

To inspect and render an output again:

```sh
.venv/bin/python tools/inspect_glb.py RUN_DIR/model.glb --no-render
node tools/render_mlx_glb.cjs RUN_DIR/model.glb RUN_DIR/model.preview.png
```

The renderer uses the isolated Playwright installation at `.deps/render-qa`
and the existing Three.js package in `studio/web/node_modules`. It launches
and closes a separate headless Chrome process.

## Four-view Monkey King result

The 21 October 2026 Studio run `e6bc56a266c1443f9e5c7f2d987c139a` used all four
original `assets/case01/001.png` through `004.png` images, seed 42, the complete
selective oQ5 bundle and mixed high-resolution compute. All inference ran on
the local Apple M5. The API was restarted during generation and reconnected
to the same detached worker without restarting its completed model stages.

[GLB](outputs/studio/e6bc56a266c1443f9e5c7f2d987c139a/prediction.glb) ·
[Four-view render](outputs/studio/e6bc56a266c1443f9e5c7f2d987c139a/work/model.preview.png) ·
[Receipt](outputs/studio/e6bc56a266c1443f9e5c7f2d987c139a/receipt.json) ·
[Validation](outputs/studio/e6bc56a266c1443f9e5c7f2d987c139a/validation.json)

- 57,233,248-byte GLB, 684,317 serialized vertices, 992,890 triangles and both
  2048 × 2048 PBR maps. All eleven textured primitives rendered successfully.
- All four submitted image pixel arrays match the original demo images.
  All thirteen stage checkpoint hashes, the final output, the published copy,
  the receipt and the actual API download agree.
- Finite geometry/normals, valid face indices and UVs within the unit square.
  Native export diagnostics report `topology_clear` and artifact readiness.
- View routing used all four images: Shape512 counts were 413/273/447/413;
  Shape1024 counts were 2197/1261/2089/1905.
- Recorded stage times total 39.95 minutes. UV creation took 880.95 seconds
  (14.68 minutes), the largest individual processing step. The recorded MLX
  allocation peak was 8.99 GB, excluding CPU allocations and other applications.
- Visual inspection covered front, three-quarter, back and side: coherent
  raised-leg pose, blue scarf, gold clothing/cap, red/gold staff and cloud base.

Studio now reads the native export event stream outside the model worker,
so long UV calls expose their operation and elapsed time even when native code
holds the worker's Python GIL. Nested `uv.*` events retain their parent atlas
stage and total timer. Regression tests cover this behavior and partial event
writes; all 29 Studio tests and five upload tests passed. Completed assets show
the actual local engine and offer a control to reuse their saved references.

## Implementation

- Preserves the shared alpha crop, black-background composite and separate
  512/1024px DINO and 448px VGGT inputs.
- Runs the verified original-model DINOv3 encoder with FP16 internal matrices
  and FP32 boundaries.
- Implements DINOv2 patch embedding and all 24 alternating VGGT frame/global
  layers, including 2D rotary positions and layer scales.
- Interleaves Stage1's 30 DiT blocks with the trained per-block DINO deltas
  and timestep-gated DiT-to-VGGT residuals.
- Runs the shape-coordinate encoder, seven shape-attention adapters, four
  cached VGGT feature levels and the DPT shape-conditioned point head.
- Routes each sparse voxel to its selected view in the ten trained routed
  layers; other layers attend to all input views. One-view runs use the
  mathematically equivalent all-zero route and omit the unused point head.
- Preserves the Euler/CFG time schedule and x0 rescaling. Sparse rescaling
  matches the source `SparseTensor` population variance; dense structure
  uses PyTorch's sample standard deviation.
- Decodes the 512 mesh, remeshes and normalizes it, and uses the original
  exact triangle-box SAT surface voxelizer for the 64-grid handoff.
- Samples the 1024 shape and texture latents, shares shape subdivisions
  with texture decoding, then remeshes, generates UVs, bakes PBR maps and
  exports a GLB with the original `(x,y,z) -> (x,z,-y)` rotation.

Weights load by stage. Full attention is retained, with query chunking to
bound temporary allocations. The checkpoint's selective oQ5 matrices stay
packed; convolution, decoder, adapter and other protected weights retain
their saved precision. Geometry-sensitive paths retain FP32 arithmetic.
By default, Shape1024 and texture use FP16 matrix products and attention,
with FP32 residuals, normalization, modulation, RoPE, activation functions
and output/guidance calculations. `--high-res-precision float32` selects
the slower reference mode. Precision must match when resuming a run with
completed high-resolution stages.

`MLX_ENABLE_TF32=0` disables the M5's lower-precision FP32 matrix products.
This runtime setting does not change the quantized checkpoint files.

The runtime uses separate deterministic NumPy random streams for each stage.
Its seed is reproducible within this runtime, but is not a promise of the
same noise samples as PyTorch's random generator.

## Validation

```sh
.venv/bin/python -m pytest tests/test_mlx_runtime.py -q
```

The complete `tests/` suite passed **31 tests** after the final run; its
JUnit record is `run_logs/mlx-case02-seed42/pytest.xml`.

Numerical tests compare the port directly with repository PyTorch
implementations: antialiased position interpolation, VGGT blocks, packed
shape attention, feature adaptation, Stage1 fusion, routed attention, view
selection, full DiT blocks, occupancy decoding and Euler guidance.
Model-bundle integrity and DINO quality
evidence are documented separately in [ONNX_DINOV3.md](ONNX_DINOV3.md).

The real two-view check uses `assets/case02/001.png` and `002.png` through
DINOv3, shape-conditioned VGGT, its DPT point head, view routing and a full
Shape512 denoiser prediction. Both views were used (805 and 834 target
voxels), and the prediction was finite. The complete point head was also
compared with the original PyTorch implementation on identical inputs:
relative L2 error was 0.000333% for points and 0.000276% for confidence.
This is a multiview conditioning check, not a complete two-view GLB run.

The bundled background remover was exercised on RGB input after deliberately
discarding the supplied alpha. Its mask achieved 0.9408 intersection-over-union
against that alpha, with finite output and a plausible foreground area.
The supplied alpha is an external segmentation reference, not a paired
unquantized RMBG model output.

```sh
.venv/bin/python tools/validate_mlx_multiview.py \
  assets/case02/001.png assets/case02/002.png --run RUN_DIR
.venv/bin/python tools/validate_mlx_rmbg.py assets/case02/001.png \
  --output RUN_DIR/rmbg-validation
```

Mixed compute was compared with FP32 using the same quantized weights,
7,459 actual handoff tokens, real image conditioning and three full
30-block predictions per model (positive conditioning at t=1 and t=0.35,
negative conditioning at t=0.35):

| Model | Minimum cosine | Maximum relative L2 error | Median prediction speedup |
| --- | ---: | ---: | ---: |
| Shape1024 | 0.999999521 | 0.0982% | 2.17× |
| Texture | 0.999999709 | 0.0764% | 2.24× |

Both passed the predefined cosine ≥ 0.99999 and relative L2 ≤ 0.5% gates.
These measure arithmetic differences within the converted model; they
are not a comparison against the original unquantized CUDA pipeline.
Reproduce with `tools/validate_mlx_flow_precision.py --run RUN_DIR
--component shape1024` (or `texture`) after the necessary stage checkpoints
exist. Raw results are in `flow-precision.json` and
`texture-flow-precision.json` in the run directory.

Native mesh remeshing/simplification is a replacement for CUDA CuMesh.
Matching its intended geometric operations does not establish bitwise CUDA
parity. Quantization quality across the complete cascade requires actual
rendered results and, for a numerical quality claim, paired source runs.

The pinned native backend still labels its MLX QEM quality tier experimental
and uses an offset double-cover UDF remesh. Its artifact and viewer checks
passed, but its production-equivalence flags remain false because it has no
UniForge3D reference profile or paired CUDA visual comparison. Those flags
are preserved in `diagnostics.json`; the separate four-view inspection does
not override them. The completed four-view Monkey King run above establishes
execution on that fixture. A broader image suite and paired complete CUDA
quality comparisons remain outstanding.

The existing immutable bundle was assembled before the runtime port. Its
conversion-time manifest and README retain that earlier runtime status;
they are kept unchanged because their hashes identify the checkpoints in
the run record. Current runtime evidence is recorded separately in this
document and `run_logs/mlx-case02-seed42/validation.json`.
