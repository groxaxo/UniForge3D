"""Durable CPU-only UniForge routes mounted in the existing port-12434 proxy.

Only GPUQ starts CUDA work. HTTP requests never hold a GPU reservation and there
is no additional broker, GPU pin, detached controller, or admission polling loop.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .gpuq_common import QUEUE_TERMINAL, atomic_json, job_folder, sha256_file

HOME = Path('/home/op/uniforge3d-work')
MAX_IMAGE = 128 * 1024**2  # Normalized RGBA PNG, up to the Studio 32 MP limit.


class Gateway:
    def __init__(self, home=HOME, queue='/home/op/.local/bin/gpuq'):
        self.home = Path(home)
        self.repo = self.home / 'UniForge3D'
        self.base = self.home / 'studio-gpuq' / 'jobs'
        self.python = self.home / 'env/bin/python'
        self.queue = str(queue)

    def command(self, identifier):
        return [str(self.python), '-u', '-m', 'studio.server.gpuq_worker',
                '--job', str(job_folder(self.base, identifier))]

    def cli(self, *args):
        return subprocess.run([self.queue, *args], check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()

    @contextmanager
    def locked(self, identifier):
        folder = job_folder(self.base, identifier)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'gateway.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield folder

    def health(self):
        encoder = self.home / 'weights/dino-hf'
        verification = json.loads((encoder / 'verification.json').read_text())
        ready = (verification.get('verified') is True and self.python.is_file()
                 and (self.repo / 'studio/server/gpuq_worker.py').is_file()
                 and (self.home / 'studio-gpuq/runtime/manifest.json').is_file()
                 and (self.home / 'studio-gpuq/runtime/tools/run_cuda_staged.py').is_file()
                 and shutil.disk_usage(self.home).free > 4 * 1024**3)
        queue = json.loads(self.cli('status', '--json'))
        cards = queue.get('gpus', [])
        return dict(connected=True, ready=ready, engine_id='cuda3090',
                    engine='CUDA · RTX 3090 on .51 · GPUQ', gpu='RTX 3090 · GPUQ',
                    encoder='Original DINOv3', min_views=1, max_views=4,
                    local_mlx_ready=False, queue='gpuq', queue_mode='per-job',
                    queue_status={'queued': len(queue.get('queued_jobs', [])),
                                  'running': len(queue.get('active_jobs', []))}, gpu_count=len(cards),
                    message='RTX 3090 on .51. GPUQ assigns one available GPU; busy jobs wait in the shared queue.')

    def put_image(self, identifier, index, data):
        if index not in range(1, 5) or not 24 <= len(data) <= MAX_IMAGE:
            raise ValueError('Choose one to four Studio reference images')
        if data[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
            raise ValueError('Expected a normalized PNG')
        width, height = struct.unpack('>II', data[16:24])
        if min(width, height) < 64 or max(width, height) > 8192 or width * height > 32_000_000:
            raise ValueError('Reference image dimensions exceed Studio limits')
        with self.locked(identifier) as folder:
            if (folder / 'request.json').exists():
                # A network retry may repeat an identical upload, but never change a queued job.
                path = folder / 'input' / f'{index:03d}.png'
                import hashlib
                if path.is_file() and sha256_file(path) == hashlib.sha256(data).hexdigest():
                    return {'sha256': sha256_file(path)}
                raise HTTPException(409, 'This job has already been submitted')
            (folder / 'input').mkdir(exist_ok=True)
            path = folder / 'input' / f'{index:03d}.png'
            temporary = path.with_suffix('.upload')
            temporary.write_bytes(data)
            temporary.replace(path)
            return {'sha256': sha256_file(path)}

    def owned(self, identifier, queue_id):
        if not re.fullmatch(r'[0-9a-f]{12}', queue_id):
            raise RuntimeError('Invalid stored GPUQ job ID')
        job = json.loads(self.cli('show', queue_id))
        command = job['command']
        if isinstance(command, str):
            command = json.loads(command)
        if (job['name'] != f'uniforge3d:{identifier}' or command != self.command(identifier)
                or Path(job['cwd']) != self.repo or job['gpu_count'] != 1):
            raise RuntimeError('GPUQ ownership check failed')
        return job

    def find_submission(self, identifier):
        # Recover the durable queue entry if the gateway exited between submit and saving its ID.
        name = f'uniforge3d:{identifier}'
        matches = [line.split()[0] for line in self.cli('list').splitlines()
                   if name in line.split()]
        if len(matches) > 1:
            raise RuntimeError('Multiple queue entries found; manual inspection required')
        if matches:
            self.owned(identifier, matches[0])
            return matches[0]
        return None

    def submit(self, identifier, payload):
        seed = payload.get('seed')
        inputs = payload.get('inputs')
        if (type(seed) is not int or not 0 <= seed <= 2**32 - 1 or not isinstance(inputs, dict)
                or not 1 <= len(inputs) <= 4
                or sorted(inputs) != [f'{i:03d}.png' for i in range(1, len(inputs) + 1)]
                or not all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                           for value in inputs.values())):
            raise ValueError('Invalid seed or ordered input hashes')
        request = {'seed': seed, 'inputs': inputs}
        with self.locked(identifier) as folder:
            if sorted(path.name for path in (folder / 'input').glob('*.png')) != sorted(inputs):
                raise ValueError('Uploaded reference count does not match the job request')
            for name, digest in inputs.items():
                if not (folder / 'input' / name).is_file() or sha256_file(folder / 'input' / name) != digest:
                    raise ValueError('Reference transfer hash mismatch')
            path = folder / 'request.json'
            if path.exists() and json.loads(path.read_text()) != request:
                raise HTTPException(409, 'This job ID belongs to a different request')
            metadata = folder / 'queue.json'
            if not metadata.exists():
                atomic_json(path, request)
                queue_id = self.find_submission(identifier)
                if queue_id is None:
                    try:
                        queue_id = self.cli('submit', '--name', f'uniforge3d:{identifier}',
                                            '--count', '1', '--cwd', str(self.repo), '--',
                                            *self.command(identifier))
                    except (subprocess.SubprocessError, OSError):
                        queue_id = self.find_submission(identifier)
                        if queue_id is None:
                            raise
                self.owned(identifier, queue_id)
                atomic_json(metadata, {'id': queue_id})
        return self.status(identifier)

    def status(self, identifier):
        folder = job_folder(self.base, identifier)
        metadata = folder / 'queue.json'
        if not metadata.is_file():
            if not (folder / 'request.json').is_file():
                raise HTTPException(404, 'Job not found')
            with self.locked(identifier):
                queue_id = self.find_submission(identifier)
                if queue_id is None:
                    raise HTTPException(503, 'GPUQ submission did not finish; retry the same job ID')
                atomic_json(metadata, {'id': queue_id})
        queue_id = json.loads(metadata.read_text())['id']
        queued = self.owned(identifier, queue_id)
        state_path = folder / 'status.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        base = dict(stage=None, completed_stages=[], queue_id=queue_id)
        if queued['status'] in {'queued', 'dispatched'}:
            return dict(base, status='waiting', message=f'GPUQ · {queued["status"]} · {queue_id}')
        if queued['status'] == 'cancelled':
            return dict(base, status='cancelled', message='Generation cancelled · GPUQ')
        if queued['status'] in {'failed', 'interrupted'}:
            return dict(base, status='failed', message='GPUQ job stopped; inspect its logs before retrying',
                        error=state.get('error') or queued.get('reason', 'GPUQ worker stopped'))
        if queued['status'] == 'completed' and state.get('status') == 'complete':
            return {**base, **state}
        if queued['status'] == 'completed':
            return dict(base, status='failed', message='Worker exited without a verified GLB')
        # A complete status file alone does not prove the admitted command has exited.
        return {**base, **state, 'status': 'running',
                'message': state.get('message', f'GPUQ · running · {queue_id}')}

    def cancel(self, identifier):
        folder = job_folder(self.base, identifier)
        with self.locked(identifier):
            metadata = json.loads((folder / 'queue.json').read_text())
            queued = self.owned(identifier, metadata['id'])
            if queued['status'] not in QUEUE_TERMINAL:
                (folder / 'cancel').touch()
                self.cli('cancel', metadata['id'])
        return self.status(identifier)


def register(app, gateway=None):
    """Called once by the existing proxy; does not start a service or load a model."""
    gateway = gateway or Gateway()
    app.state.uniforge_gateway = gateway

    def boundary(request):
        if request.headers.get('x-uniforge-studio') != '1' or request.headers.get('origin'):
            raise HTTPException(403, 'Use a Studio backend connection to submit jobs')

    def invoke(function, *args):
        try:
            return function(*args)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            raise HTTPException(503, 'UniForge GPUQ backend unavailable; inspect the proxy log') from error

    @app.get('/v1/3d/health')
    def health():
        try:
            value = gateway.health()
            return JSONResponse(value, status_code=200 if value['ready'] else 503)
        except (OSError, ValueError, subprocess.SubprocessError):
            return JSONResponse({'connected': True, 'ready': False,
                                 'message': 'UniForge GPUQ setup needs attention'}, status_code=503)

    @app.put('/v1/3d/jobs/{identifier}/images/{index}')
    async def upload(identifier: str, index: int, request: Request):
        boundary(request)
        try:
            job_folder(gateway.base, identifier)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if index not in range(1, 5):
            raise HTTPException(400, 'Invalid reference index')
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_IMAGE:
                raise HTTPException(413, 'Normalized reference image is too large')
            data.extend(chunk)
        return await run_in_threadpool(invoke, gateway.put_image, identifier, index, bytes(data))

    @app.post('/v1/3d/jobs/{identifier}')
    async def submit(identifier: str, request: Request):
        boundary(request)
        # Metadata is tiny; do not allow an unbounded JSON request on this route.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 4096:
                raise HTTPException(413, 'Job metadata is too large')
            body.extend(chunk)
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError('Expected a JSON job request')
        except ValueError as error:
            raise HTTPException(400, 'Invalid job metadata') from error
        return await run_in_threadpool(invoke, gateway.submit, identifier, payload)

    @app.get('/v1/3d/jobs/{identifier}')
    def status(identifier: str):
        return invoke(gateway.status, identifier)

    @app.post('/v1/3d/jobs/{identifier}/cancel')
    def cancel(identifier: str, request: Request):
        boundary(request)
        return invoke(gateway.cancel, identifier)

    @app.get('/v1/3d/jobs/{identifier}/{asset}')
    def download(identifier: str, asset: str):
        if asset not in {'model', 'receipt'}:
            raise HTTPException(404, 'Asset not found')
        if invoke(gateway.status, identifier)['status'] != 'complete':
            raise HTTPException(409, 'The asset is not complete')
        folder = job_folder(gateway.base, identifier)
        path = folder / ('work/prediction.glb' if asset == 'model' else 'receipt.json')
        return FileResponse(path, media_type='model/gltf-binary' if asset == 'model' else 'application/json')
