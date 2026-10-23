# Studio backends and GPUQ

Choose an engine from the **server icon** in the iOS toolbar or the **UniForge ·
engine** menu in Android's top bar. Version **0.2.0** of both apps offers all three
choices and saves the selection across relaunches. Changing engines opens that
engine's independent queue; it does not move or cancel existing jobs.

| App menu | Studio address | Compute and admission |
|---|---|---|
| Mac · Local MLX | `100.85.200.56:7861` | Apple Silicon, selective oQ5 bundle |
| RTX 3060 · .54 | `100.85.200.56:7863` | Existing private RTX 3060 queue |
| RTX 3090 · .51 · GPUQ | `100.85.200.56:7864` | Existing `.51:12434` proxy, one flexible GPUQ GPU per generation |

Keep Tailscale and the Mac services available. The mobile apps select and control
the existing backends; model inference runs on those computers. **Preserve face**
continues to use the existing verified CPU correction step on the Mac after the
GLB returns, with the original geometry/texture retained when matching fails.

## RTX 3090 route

The Mac queue's `Cuda3090Bridge` uses only
`http://100.85.200.51:12434/v1/3d`. It transfers one to four normalized PNGs,
compares their SHA256 values, submits their seed/input manifest, polls status,
and verifies the returned GLB before publishing it. No direct CUDA/SSH launch is
performed by that bridge.

The existing proxy mounts the CPU-only `gpuq_gateway` routes and advertises
`uniforge3d` as an **image-to-3D** model in `/v1/models`. It is not a chat model.
The gateway stores requests under
`/home/op/uniforge3d-work/studio-gpuq/jobs/<studio-id>/` and submits exactly one:

```sh
/home/op/.local/bin/gpuq submit --name uniforge3d:STUDIO_ID --count 1 \
  --cwd /home/op/uniforge3d-work/UniForge3D -- \
  /home/op/uniforge3d-work/env/bin/python -u -m studio.server.gpuq_worker \
  --job /home/op/uniforge3d-work/studio-gpuq/jobs/STUDIO_ID
```

`--count 1` leaves GPU placement to the shared scheduler. The worker requires
GPUQ ownership, preserves assigned `CUDA_VISIBLE_DEVICES`, uses logical `cuda:0`,
and keeps each stage in the admitted foreground process group. No additional
reservation wraps HTTP calls. The gateway uses durable per-job queue admission;
it is CPU-only and does not reserve a long-lived model-server GPU.

The shared queue's existing admission rules, preferred placement, other jobs and
**300-second GPU-idle watchdog** remain in force. There is no manual GPU eviction,
VRAM clearing, physical-device pin or automatic retry of failed/interrupted jobs.
Queue IDs and stage checkpoints are retained. Inspect `gpuq show ID` and
`gpuq logs ID` before deciding to retry a failed run.

Submission is idempotent for a Studio ID and immutable seed/input manifest.
After a proxy restart or an uncertain submit response, the gateway reconciles its
unique job name and verifies the saved command before adopting it. Cancellation
also verifies command, working directory, name and one-GPU ownership before
calling `gpuq cancel`; it cannot signal an unrelated job. Completed status and
downloads require both verified worker output and GPUQ command completion.

The CUDA path retains source precision, the original verified DINOv3 encoder,
1024 geometry, 2048 textures and the configured full sampling schedule. It uses a
separate immutable runtime under `studio-gpuq/runtimes/<manifest-id>/`; `runtime`
points at the active version. Workers resolve that pointer once and verify all
88 source-file hashes before starting. The older four-view `.51` runner and
existing `.54` source tree are left independent.
The Mac's legacy `UNIFORGE_ENGINE=cuda` direct launcher now rejects new starts
before SSH or CUDA work; use port 7864 for `.51`. Existing saved legacy results
remain readable. The dedicated `.54` bridge keeps its existing behavior.

## Operations

Start the Mac's independent persistent CPU queue:

```sh
./tools/studio_3090_service.sh install
./tools/studio_3090_service.sh status
```

It installs `com.groxaxo.uniforge-studio-3090` and writes
`run_logs/studio-3090.log`. `stop` stops that Mac API only; GPUQ owns admitted
remote commands and their durable status. Queued/generated assets stay in
`outputs/studio-3090/`.

Passive checks do not load a CUDA model:

```sh
curl http://100.85.200.51:12434/v1/3d/health
curl http://100.85.200.51:12434/v1/models
curl http://100.85.200.56:7864/api/health
```

Proxy routes use generated 32-character Studio IDs:

- `PUT /v1/3d/jobs/{id}/images/{1..4}`: normalized PNG bytes.
- `POST /v1/3d/jobs/{id}`: JSON `seed` and ordered filename-to-SHA256 `inputs`.
- `GET /v1/3d/jobs/{id}`: queued/running/complete/failed/cancelled status.
- `POST /v1/3d/jobs/{id}/cancel`: cancel this verified GPUQ job.
- `GET /v1/3d/jobs/{id}/model` or `/receipt`: completed output/provenance.

Writes require the existing `x-uniforge-studio: 1` header and reject browser
Origin requests; the Mac bridge performs these calls. This follows the private
Studio boundary and does not replace Tailscale access controls.

## Deployment and checks

`python3 tools/package_cuda_runtime.py` creates the source-only runtime archive,
manifest ID and gateway bootstrap archive under `outputs/deployment/3090/`.
Deploy the runtime into a new matching `studio-gpuq/runtimes/<id>` directory,
verify its hashes and atomically switch the `runtime` symlink. Deploy the gateway
modules into `.51`'s UniForge checkout. Existing source weights/environment are
reused; the archives contain no weights, signing material or credentials.

On `.51`, integrate with the existing proxy:

```sh
python3 /home/op/uniforge3d-work/UniForge3D/tools/configure_uniforge_proxy.py \
  --proxy-root /home/op/ULTIMATE/NAMEOFMODEL \
  --runtime-repo /home/op/uniforge3d-work/UniForge3D
```

The helper validates its exact source anchors, makes private backups, preserves
file permissions and all other model families, and refuses a conflicting
UniForge configuration. Review its diff and confirm zero active proxy requests
and proxy-owned GPU jobs before restarting `qwen36-multi-proxy.service`. Do not
restart GPUQ workers or unrelated inference services for this deployment.

```sh
studio/.venv/bin/python -m pytest studio/tests/test_gpuq_backend.py studio/tests/test_api.py -q
npm --prefix studio/web run build
./tools/test_ios.sh
./tools/build_android.sh
```

GPUQ tests cover idempotent/recovered submission, transfer hashes, immutable
inputs, cancellation ownership, interrupted jobs, publication gating and proxy
configuration preservation. Live iOS tests include selecting `.51` and keeping
that choice after relaunch. An actual CUDA test is recorded separately from
those tests: a queued job alone does not prove inference completion or quality.

## Validated deployment, 24 October 2026

The live Studio test `a887bec93b83476883d4892c5b368be3` used public
`assets/case01/001.png`, one view, seed 42 and the original DINOv3 encoder.
GPUQ job `43544b640cbc` completed with exit code 0, a flexible one-GPU request,
no pinned GPU list and physical GPU 1 assigned by the scheduler. All seven
stages completed; their timers sum to 273.90 seconds, excluding startup/transfer
and optional face processing.

The returned GLB has **32,231,892 bytes**, **613,480 vertices**, **939,467
triangles**, 2048px textures and SHA256
`1cde1dad5b7ea23cbf626cc9cdb93348df7d410980356786346bd10fc866d9c0`.
The Mac download matched that hash and the new asset rendered in Chrome. Face
matching was explicitly **skipped** for this reference, retaining the original
texture; this run is not evidence of a newly corrected face.

The GPUQ suite's 13 cases and the API regression suite passed. iOS passed 15
unit tests and 5 live simulator UI tests. The `.51` selection/relaunch test also
passed on the physical iPhone. Android passed 4 unit tests, release lint and
signature verification; the physical Moto G54 displayed all three menu choices,
selected `.51`, and retained it after force-stop/relaunch. Both apps were
updated to version 0.2.0 using the existing signing identities.

The proxy retained all 107 pre-existing model-family entries and still reported
GPUQ enabled with both idle thresholds at 300 seconds. Machine-local receipts,
GLB, manifest, deployment backup location and device/test evidence are retained
under `outputs/deployment/3090/`, `outputs/ios/` and `outputs/android/` (ignored by
Git). They contain no published signing material.
