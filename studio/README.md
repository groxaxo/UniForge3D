# UniForge Studio

A local React asset studio with four-view uploads, ordered references, a seed,
a persistent generation queue, seven real stage updates, cancellation, textured
and clay WebGL previews, and GLB / provenance downloads.

Run from the repository root:

```sh
./tools/studio_service.sh install
```

On this Mac, this installs a user login service that keeps the local server
running after Terminal closes and restarts it if it exits. Double-click
`Open UniForge Studio.command` in the repository to start the service and open
the browser. Repeating `start` keeps an already running server and its jobs.
Logs are written to `run_logs/studio-server.log`.

```sh
./tools/studio_service.sh status
./tools/studio_service.sh stop       # stop for the current login session
./tools/studio_service.sh start
./tools/studio_service.sh uninstall  # remove automatic startup; retain assets
```

For foreground development, stop the service and run `./tools/start_studio.sh`.
Only one Studio server can use the queue at a time.

Open **http://127.0.0.1:7861** on this Mac or **http://100.85.200.56:7861** from another Tailscale device. The production frontend and API share this server, bound to **0.0.0.0:7861**. The example loads the golden figure and the
completed local MLX GLB from `run_logs/mlx-case02-seed42`. Remove the example
image to upload your own; view 01 is the anchor and the left-arrow controls
reorder subsequent views. Use one to four views of the same object. The offered
pipeline is 1024 geometry / 2048 texture.

An **Optional** menu sits under the seed. It defaults to **None**. Installed
SAM 3D Body checkpoints from `Comfy-Org/sam-3d-body` appear there and stay
optional: choosing one records a **reference-only** `optional_model` on the job
and its downloadable receipt. **SAM 3D Body is not executed**; the choice does
not change generation. The UI states this before submission, and receipt
provenance includes `executed: false` plus the expected checkpoint SHA256.
Weights live in `weights/sam-3d-body/detection/`. A checkpoint is offered only
after its safetensors header, size and complete SHA256 match the published
[pinned source](https://huggingface.co/Comfy-Org/sam-3d-body/tree/7c974aea58e062252a6cb5b6f9b7f639d545b729/detection).
Verification is cached by file identity/size/timestamps; changing a file
invalidates the cache. The same weight tree is kept on the GPU host at
`/home/op/uniforge3d-work/weights/sam-3d-body/`; it is not needed to execute this
metadata-only selection.

Completed assets include a **Reuse reference images** control beside their
download actions. It restores the saved views, name and seed into the composer
without starting another job. Completed MLX assets are labelled **Local MLX**.

Reference selection supports the HTTP network address as well as localhost.
A percentage bar tracks reading the selected files; submitting shows actual
upload progress from 0–100%. At 100% sent, Studio shows that the server is
checking the images before the job appears in the queue. Failed uploads show
an error and allow retrying without selecting the files again.

Upload request regression tests use Node's built-in runner:

```sh
node --test studio/web/tests/upload.test.js
```

## Engine

New jobs run **locally on Apple Silicon**, using the complete selective oMLX
oQ5 bundle and the tested original DINOv3 encoder. The API environment remains
isolated in `studio/.venv`; each model worker uses the repository's `.venv` and
`uniforge3d_mlx.Pipeline`. See [the runtime evidence](../MLX_RUNTIME.md).

One job runs at a time. Allow about 30 minutes for a single image on this M5;
additional views can take longer. Unload other large GPU models before starting
a generation. The UI reports actual stage and sampling-step progress, including
mesh/UV processing and export in the final stage. Native export events now
identify mesh optimization, UV-atlas creation, texture baking and GLB writing;
the elapsed time updates while a long native operation is still running.

Local inputs, checkpoints, worker logs and results are retained under
`outputs/studio/<job-id>/`. A detached worker survives an API restart, and Studio
reconnects to its status. Cancel stops only the matching job's process group;
PID ownership is checked against the job path and a unique launch token before
signalling. If a worker exits unexpectedly, the job becomes failed and its
diagnostic log and completed checkpoints remain available.

Completed GLBs are parsed and hashed, copied to the download location, and
verified again before the API marks the asset ready. The receipt includes the
input hashes, seed, bundle identity, runtime stage record and output hash.

### Legacy private CUDA engine

The default engine is MLX. `UNIFORGE_ENGINE=cuda ./tools/start_studio.sh` selects
the older remote engine for foreground operation after stopping the login
service. Existing jobs retain their original engine when Studio restarts.

The UI/API runs on this Mac. Reference images are sent over SSH to your private
Tailscale host `op@100.85.200.51`. Generation uses CUDA source weights and the
normalized SoftEq encoder (FP16 storage, FP32 compute in the original loader).
This backend requires four views and remains independent of the local MLX path.

The Studio worker uses the spare RTX 3090 UUID
`GPU-828df6fd-3fd0-ed25-0b2b-2b6d9d8dca47`; an occupied GPU is left alone and
the job waits. It reuses `/home/op/uniforge3d-work/UniForge3D/tools/run_cuda_staged.py`
without changing the existing runner or baseline encoder. Its files live only
in `/home/op/uniforge3d-work/studio/`. All existing services remain independent.

Model: `ashiq24/softeq-dinov3-vitl16-pretrain-lvd1689m-ade-seg-c360-s0.1-sp0.8`,
revision `f853065890a0df7869f9b5267361cdc0ac139fee`. The worker verifies the
prepared encoder SHA256 before inference, records converted input hashes and
seed, and the Mac verifies the downloaded GLB hash before marking a job ready.
SoftEq is an experimental replacement for the trained conditioning encoder;
successful export does not establish equivalent asset quality. Inspect the
mesh before using it.

Jobs and returned assets are saved in `outputs/studio/<job-id>/`. Remote stage
checkpoints and logs are retained in `studio/jobs/<job-id>/` for diagnosis and
resume; there is no automatic cleanup. Keep both disks available for new jobs.
On API restart queued jobs resume, and an already started remote job is
reconnected. A failed connection eventually requests cancellation of its own
remote job and records the failure. Cancellation stops only that job's child
process group. Each stage has a 30 minute timeout.

The server binds to all interfaces and checks the configured host/origin plus
the Studio API header. Allowed hosts include localhost, `100.85.200.56`,
`172.20.10.2` and `op-2.local`. Set `UNIFORGE_ALLOWED_HOSTS` to a comma-separated
host list when using other addresses. Requests from unrelated browser origins
remain rejected.

## Rebuild

Dependencies are isolated from the model conversion environment:

```sh
uv venv --python .venv/bin/python studio/.venv
uv pip install --python studio/.venv/bin/python -r requirements-studio.txt
npm --prefix studio/web ci
npm --prefix studio/web run build
```

For frontend development, run the API plus `npm --prefix studio/web run dev`.
Vite proxies `/api` to port 7861. The production server serves the built assets
itself, with no CDN or external model/API requests in the browser.

The deployed remote script is `studio/server/remote_runner.py`. Update it only
when there are no running Studio jobs. The prepared encoder's config,
`model.safetensors` and `precision-verification.json` belong in the remote
`studio/encoder/` folder. Its expected file SHA256 is pinned in the worker;
the original `weights/dino-hf` is preserved.

## Validation status

The local MLX integration has API tests for one, two and four images, queue
validation, worker crash/cancellation handling, PID ownership and hash-verified
GLB publication. The frontend production build passes. The runtime's complete
single-image output and numerical validation are linked above. The frontend
also shows that completed local result as its default interactive example.

The following older notes apply to the **legacy CUDA backend**, tested on
20 October 2026:

The frontend build and eleven API tests pass. Browser checks cover desktop and
mobile layouts, image removal/upload, material switching and camera reset.
The example GLB downloaded through the API has the same SHA256 as its source.
The prepared SoftEq encoder has been copied to the private host and its full
SHA256 matches the pinned value.

A **new SoftEq image-to-3D run is not verified**: the private host began reporting
`Problem with GPU 0000:02:00.0: Unknown Error`, and a separate PyTorch probe
returned CUDA unavailable / zero devices. Studio displays this state and disables
Generate. Restore the host GPU driver/hardware access before testing a new job.
The UI checks again periodically. The original example is a previous CUDA run
with the original encoder; it is not evidence of SoftEq generation quality.

The Mac login service was verified with automatic restart after termination,
an idempotent start, the example preview and a hash-matched 34.1 MB GLB download.
The API also rejects new jobs with HTTP 503 and the engine status when the private
GPU is unavailable, before saving or transferring a job. The eleven API tests
and twelve conversion tests pass.

To run the API tests in the isolated environment:

```sh
uv pip install --python studio/.venv/bin/python pytest==9.1.1 httpx==0.28.1
studio/.venv/bin/python -m pytest studio/tests/test_api.py -q
```

## Private RTX 3060 queue (.54)

`./tools/start_studio_3060.sh` opens a separate queue at
**http://127.0.0.1:7863** and **http://100.85.200.56:7863**, bound to **0.0.0.0:7863**. The local MLX queue on 7861 remains independent.
CUDA generation uses GPU `GPU-72f3cda5-5f26-bed3-6093-0608adb365e3`
on `op@100.85.200.54`, the original verified DINOv3 encoder, source precision,
one to four ordered views, and the complete 12/20/20/20 sampling schedule.
Jobs are stored locally in `outputs/studio-3060/` and remotely in
`/home/op/uniforge3d-work/studio-3060/jobs/`.

The RTX 3060 host reuses the existing Python 3.10 CUDA environment and weights
through a **read-only** SSHFS mount from `.51`. Three shared backbone files also
have SHA256-verified copies on `.54` to reduce repeated network reads;
`/home/op/uniforge3d-work/cache-receipt.json` records them. No CUDA computation runs on `.51`.
The mount is `/home/op/uniforge3d-work/source-ro`, with `env` and `weights`
symlinks in its parent. To restore the mount on `.54` after a reboot:

```sh
sshfs op@100.85.200.51:/home/op/uniforge3d-work \
  /home/op/uniforge3d-work/source-ro \
  -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

Generation is serialized and waits for at least 10 GiB of GPU headroom. It leaves
the existing small embedding service running, uses expandable CUDA allocation,
and loads each stage in a separate process. Each output receipt records the
actual GPU UUID/name, host, input hashes, encoder hash and per-stage CUDA memory
measurements. The Mac parses and verifies the returned GLB hash before publication.
The hashing helper supports Python 3.10 without `hashlib.file_digest`.

The 2026-10-21 input sheets and integer pixel crop rectangles are recorded in
`run_logs/downloads-20261021/crops.json`. The original downloads are retained;
full-resolution PNG crops are in `~/Downloads/UniForge3D-Cropped-20261021/`.

The staged CUDA runner defers mesh hole filling until both VAE decoders are
offloaded and their sparse convolution/subdivision caches are released. This
bounds final mesh-processing memory on the 12 GB card while preserving decoded
features, geometry resolution, textures and sampling steps. The shared decoder
helper retains its original behavior by default; the staged runner requests
`fill_holes=False` and fills the mesh after releasing decoder resources.

A failed RTX 3060 job can recover retained checkpoints with
`POST /api/jobs/{id}/resume` using the Studio mutation header. Recovery requires a
failed, unlocked remote controller and archives its failure logs. It retains
completed stages and queues the original asset before newer waiting jobs, after
the active generation. Restart reconnects an already active worker first.

Large creatures can exceed the card's topology-processing memory even after
model offload. The 12 GB staged path therefore caps intermediate CUDA meshes at
4,000,000 faces using CPU quadric simplification, both before initial topology
work and after 1024 remeshing. The final target remains 1,000,000 faces with 2048
PBR maps. This adds an intermediate geometry approximation; inspect thin details
in the saved previews. The higher-memory default CUDA path has no such cap.

The pinned CPU helper is isolated from the read-only shared Python environment:

```sh
/home/op/uniforge3d-work/env/bin/python -m pip install \
  --target /home/op/uniforge3d-work/cpu-mesh-deps --no-deps --no-cache-dir \
  fast-simplification==0.1.13
```

The worker retains an atomic `work/decoded.pt` containing raw decoded vertices,
faces, PBR voxel attributes, RNG states and input/config/latent fingerprints.
Further export retries can reuse it without repeating either VAE or sampling.
The configured Context7 service answered its liveness probe but its documentation
quota was exhausted; the helper API was checked against the installed pinned
package source and exercised on the private host before use.

## Preserve face

The generation form includes an optional **Preserve face** checkbox. It runs a
local CPU face-photo correction after either MLX or CUDA generation. Add a clear,
matching face close-up, or leave that field empty to try the first reference view.
The option and normalized reference are stored with the job and restored when
reusing its references. The API rejects the option before queueing if the local
worker/models are missing.

The pinned `tools/ComfyUI-GLB-Face-Retexture` v1.4.0 installation needs its own
Python 3.12 `.venv` and downloaded models; see its installer and
[the compatibility instructions](../FACE_COMPATIBILITY.md). Install it separately
from both the Studio API and MLX environments. The Studio queue runs the tested
same-material chunk converter first, then the face worker, with skin recolor
left off. There is no cloud request or repeated neural 3D generation.

Publication requires unchanged geometry, original embedded-image preservation,
equal triangle counts, zero changed protected-body pixels and a valid output GLB.
Applied results say **Face preserved from your reference photo**. If the model or
photo cannot be matched, Studio keeps the generated GLB and displays **Face
correction could not be applied. Original face texture kept.** It never labels a
failed match as a preserved face. This corrects visible texture, not silhouette,
missing anatomy or expression. A recognizable matching face is required in the
photo and rendered source.

The original generation is retained as `generation.glb`. Corrected candidates,
comparison images, verification proofs and logs are under
`face-preservation/<attempt-id>/`. The download receipt wraps the original
generation receipt and records the post-process result and actual published GLB
hash. Cancellation stops only that face worker's process group; API restart
leaves the job resumable. The checkbox is off by default.
