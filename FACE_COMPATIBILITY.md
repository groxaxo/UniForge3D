# MLX GLB compatibility for the face-retexture tool

`tools/glb_face_compat.py` converts UniForge3D's same-material primitive chunks
into one indexed primitive accepted by the pinned
[ComfyUI-GLB-Face-Retexture v1.4.0](https://github.com/KaletoAI/ComfyUI-GLB-Face-Retexture/tree/832894ec8ae991eb1cd5483cd80352cd38c9c76a).
This is a separate post-processing step; the upstream tool remains unchanged.
The pinned tool is tracked as a Git submodule. Initialize it after a normal
clone (or clone this repository with `--recurse-submodules`):

```sh
git submodule update --init --recursive tools/ComfyUI-GLB-Face-Retexture
```

Its isolated `.venv` and downloaded `models/` remain local and are not committed.

## Convert, then correct the face texture

```sh
.venv/bin/python tools/glb_face_compat.py original.glb compatible.glb \
  --proof compatibility.json
```

The converter requires NumPy, already included in `requirements-mlx.txt`. Use a
new destination and proof path; it refuses to overwrite the source or existing
outputs. Every chunk must share the same material index and float32 attribute
layout. Only one static mesh instance with triangle primitives is supported.
Positions, normals, UVs and optional tangents are concatenated byte-for-byte.
Vertices are not welded and triangles are not simplified or reordered. Indices
are rebased to unsigned 32-bit; original viewer bounds metadata is preserved.
Different materials, inconsistent layouts, morph targets and animation/skin
inputs are refused. This does not merge texture atlases or repair facial geometry.

Prepare the pinned face tool's isolated Python 3.12 worker as described in its
README. Then run its CLI on the compatible file and a matching RGB face photo:

```sh
FACE_TOOL=/path/to/ComfyUI-GLB-Face-Retexture
"$FACE_TOOL/.venv/bin/python" "$FACE_TOOL/worker.py" \
  --source-glb compatible.glb --reference-image face.png \
  --output-glb corrected.glb --comparison-png comparison.png \
  --proof-json face-proof.json --model-dir "$FACE_TOOL/models" \
  --strength 1 --feather 0.18 --light-balance 1 --tone-match 1 \
  --brightness 1 --skin-recolor 1
```

The color controls are starting points for inspection. Skin recolor can affect
clothing that resembles skin; set `--skin-recolor 0` when inappropriate. A detectable
face must be visible in the source registration view and photo. Use `--yaw` to
adjust registration orientation. Opaque embedded PNG/JPEG/WebP textures and static
Y-up/+Z-front assets are the upstream supported input contract. Its own input
validation still runs after conversion. The operation projects a face texture;
it cannot reconstruct hidden anatomy, expression, silhouette or missing geometry.

## Validation on 23 October 2026

The original four-view MLX Monkey King GLB has 11 primitives, 684,317 serialized
vertices and 992,890 triangles. The converter produced one primitive with exactly
the same vertex attribute bytes and ordered triangles after index rebasing.
Materials, textures, image bytes, nodes and the original binary prefix were
preserved. The converted model passed the real upstream `read_source()` gate.

| Artifact | SHA-256 / bytes |
| --- | --- |
| Original Monkey King | `38674ac1f6c6a2b2fd55dae1252bb88c0296b1fd4d016be73209f2c536eaa3da` / 57,233,248 |
| Compatible Monkey King | `6d2b75102e07207a58b9e78d487b72ed0a18f69c503f0a92b41a2a334f228ce0` / 91,046,008 |

The file grows because new combined accessors are appended while original binary
bytes remain retained for audit. No quantization or model generation is repeated.
Four independent Three.js views rendered successfully with identical bounds.
The 1024×1084 screenshots differed at five pixels, with maximum channel difference
3/255; screenshot pixel identity is not claimed. Vertex/UV/normal data and texture
preservation are checked directly, separately from this renderer observation.
Visual inspection covered front, three-quarter, back and side.

A controlled public NASA portrait-grid fixture was split into two primitives,
refused by upstream before conversion, converted, then run through full MediaPipe
face registration, skin recolor and texture projection. Compatibility preserved
the fixture's CPU albedo render pixel-exactly. The corrected export retained all
1,152 triangles, reported unchanged geometry and original embedded images, and
changed **zero protected-body render pixels**; 19,621 head pixels changed.
This establishes complete conversion-to-face-tool execution on a detectable face.
The Monkey King's format compatibility is verified separately; detection and photo
projection on its stylized face are not claimed by that fixture result.

Machine-readable results: [`docs/face-compatibility-validation.json`](docs/face-compatibility-validation.json).

## Repeat the checks

```sh
.venv/bin/python -m pytest tests/test_glb_face_compat.py -q
# Use the face tool's private worker interpreter, not the MLX environment:
FACE_TOOL=/path/to/ComfyUI-GLB-Face-Retexture
"$FACE_TOOL/.venv/bin/python" tools/validate_glb_face_compat.py \
  --face-tool "$FACE_TOOL" --output new-validation-directory \
  --model-glb compatible.glb
```

The adapter has ten regression cases, including exact attribute/index retention,
interleaved accessors, viewer-bound rounding, malformed indices, mismatched
materials/layouts, morph/instance refusal and source/output protection. The pinned
upstream face-tool suite separately passed all 49 tests. No source GLBs, user face
photos or model weights are committed with this compatibility step.
