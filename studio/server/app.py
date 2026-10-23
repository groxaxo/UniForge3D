"""Studio API and frontend with MLX generation and legacy private CUDA jobs."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .remote_runner import glb_metadata
from .local_bridge import LocalBridge
from .face_preservation import face_readiness, preserve_face, FaceCancelled

ROOT = Path(__file__).resolve().parents[2]
PHASES = ['preprocess', 'dino', 'structure', 'shape512', 'shape1024', 'texture', 'decode']
TERMINAL = {'complete', 'failed', 'cancelled'}
MAX_FILE = 15 * 1024**2


@dataclass(frozen=True)
class Settings:
    data: Path = ROOT / 'outputs/studio'
    host: str = os.environ.get('UNIFORGE_SSH_HOST', 'op@100.85.200.51')
    remote: str = '/home/op/uniforge3d-work/studio'
    python: str = '/home/op/uniforge3d-work/env/bin/python'
    engine: str = os.environ.get('UNIFORGE_ENGINE', 'mlx')
    port: int = 7861
    allowed_hosts: tuple[str, ...] = tuple(host.strip() for host in os.environ.get(
        'UNIFORGE_ALLOWED_HOSTS',
        '127.0.0.1,localhost,testserver,100.85.200.56,172.20.10.2,op-2.local',
    ).split(',') if host.strip())


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def normalize_image(data):
    if len(data) > MAX_FILE:
        raise ValueError('Each image must be 15 MB or smaller.')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('Please choose PNG, JPG or WebP images.')
            if min(image.size) < 64 or max(image.size) > 8192 or image.width * image.height > 32_000_000:
                raise ValueError('Images must be at least 64 pixels and no larger than 8192 pixels / 32 megapixels.')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Animated images are not supported.')
            image.load()
            image = ImageOps.exif_transpose(image)
            image = image.convert('RGBA' if image.mode in {'RGBA', 'LA', 'P'} else 'RGB')
            image.info.clear()
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('This file could not be read as a supported image.') from exc


def image_response(path, thumbnail):
    if not thumbnail:
        return FileResponse(path, media_type='image/png')
    with Image.open(path) as image:
        image = image.convert('RGBA')
        bounds = image.getbbox()
        if bounds:
            image = image.crop(bounds)
        image.thumbnail((320, 320))
        canvas = Image.new('RGBA', (360, 360), (255, 255, 255, 0))
        canvas.alpha_composite(image, ((360 - image.width) // 2, (360 - image.height) // 2))
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
    return Response(buffer.getvalue(), media_type='image/png', headers={'Cache-Control': 'private, max-age=3600'})


class RemoteBridge:
    engine = 'cuda'
    def __init__(self, settings):
        self.settings = settings

    def ssh(self, command, timeout=30, **kwargs):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', self.settings.host, command],
            check=True, timeout=timeout, **kwargs)

    def read_status(self, job_id):
        path = f'{self.settings.remote}/jobs/{job_id}/status.json'
        result = self.ssh(f'cat {shlex.quote(path)}', capture_output=True, text=True)
        return json.loads(result.stdout)

    def health(self):
        if self.engine == 'cuda':
            return {'connected': True, 'ready': False, 'engine_id': 'cuda',
                    'local_mlx_ready': False,
                    'message': 'Select RTX 3090 · .51 · GPUQ on Studio port 7864. The legacy direct CUDA launcher is retired.'}
        remote = self.settings.remote
        code = f'''import json, shutil, subprocess
from pathlib import Path
p = Path({remote!r})
result = subprocess.run(['nvidia-smi', '-i', 'GPU-828df6fd-3fd0-ed25-0b2b-2b6d9d8dca47',
    '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
g = result.stdout.strip().split(',')
health = {{'connected': True, 'gpu': 'NVIDIA RTX 3090', 'used_mib': 0, 'total_mib': 24576,
    'encoder_ready': (p/'encoder/precision-verification.json').exists() and (p/'encoder/model.safetensors').stat().st_size == 606300344,
    'worker_ready': (p/'remote_runner.py').exists(), 'free_bytes': shutil.disk_usage(p).free}}
if result.returncode == 0 and len(g) == 3 and g[1].strip().isdigit() and g[2].strip().isdigit():
    health.update(gpu=g[0].strip(), used_mib=int(g[1]), total_mib=int(g[2]))
else:
    health.update(gpu_error=(result.stdout + result.stderr).strip()[:400],
        message='Private host is online, but its NVIDIA GPU is unavailable. Restore GPU access on the host.')
print(json.dumps(health))
'''
        result = self.ssh(f'{shlex.quote(self.settings.python)} -c {shlex.quote(code)}',
                          timeout=15, capture_output=True, text=True)
        health = json.loads(result.stdout)
        health['ready'] = health['encoder_ready'] and health['worker_ready'] and health['free_bytes'] > 4 * 1024**3 and not health.get('gpu_error')
        health['engine'] = 'CUDA · source precision + SoftEq FP16 encoder'
        health.update(engine_id='cuda', encoder='SoftEq DINOv3', min_views=4, max_views=4)
        health['local_mlx_ready'] = False
        return health

    def start(self, job_id, directory, seed):
        if self.engine == 'cuda':
            raise RuntimeError('Direct CUDA launch is disabled. Use the RTX 3090 GPUQ backend on port 7864.')
        remote_job = f'{self.settings.remote}/jobs/{job_id}'
        self.ssh(f'mkdir -p {shlex.quote(remote_job + "/input")}')
        # Fixed generated names: user filenames and asset titles never enter shell commands.
        for i in range(1, 5):
            path = directory / 'input' / f'{i:03d}.png'
            with path.open('rb') as stream:
                self.ssh(f'cat > {shlex.quote(remote_job + "/input/" + path.name)}', stdin=stream, timeout=120,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        inputs = {path.name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                  for path in sorted((directory / 'input').glob('*.png'))}
        payload = json.dumps({'seed': seed, 'inputs': inputs}).encode()
        self.ssh(f'cat > {shlex.quote(remote_job + "/request.json")}', input=payload, capture_output=True)
        command = (f'nohup {shlex.quote(self.settings.python)} -u {shlex.quote(self.settings.remote + "/remote_runner.py")}'
                   f' --job {shlex.quote(remote_job)} > {shlex.quote(remote_job + "/controller.log")} 2>&1 < /dev/null &')
        self.ssh(command, capture_output=True)

    def cancel(self, job_id):
        self.ssh(f'touch {shlex.quote(self.settings.remote + "/jobs/" + job_id + "/cancel")}', capture_output=True)

    def fetch(self, job_id, directory, metadata):
        remote_job = f'{self.settings.remote}/jobs/{job_id}'
        temp = directory / 'prediction.download'
        with temp.open('wb') as stream:
            self.ssh(f'cat {shlex.quote(remote_job + "/work/prediction.glb")}', timeout=180,
                     stdout=stream, stderr=subprocess.PIPE)
        checked = glb_metadata(temp)
        if checked['sha256'] != metadata['sha256'] or checked['bytes'] != metadata['bytes']:
            temp.unlink(missing_ok=True)
            raise RuntimeError('The downloaded GLB did not match the GPU output. Please retry.')
        temp.replace(directory / 'prediction.glb')
        result = self.ssh(f'cat {shlex.quote(remote_job + "/receipt.json")}', capture_output=True, text=True)
        atomic_json(directory / 'receipt.json', json.loads(result.stdout))
        return checked


class JobStore:
    def __init__(self, settings, bridge):
        self.settings, self.bridge = settings, bridge
        self.engine = getattr(bridge, 'engine', 'cuda')
        self.lock = threading.RLock()
        self.jobs = {}
        self.pending = queue.PriorityQueue()
        self.stop = threading.Event()
        self.health_cache = (0, {'connected': False, 'ready': False, 'message': 'Checking generation engine…'})
        self.health_lock = threading.Lock()
        settings.data.mkdir(parents=True, exist_ok=True)
        for path in settings.data.glob('*/job.json'):
            try:
                data = json.loads(path.read_text())
                self.jobs[data['id']] = data
            except (OSError, ValueError, KeyError):
                continue

    def start(self):
        self.file_lock = (self.settings.data / 'server.lock').open('w')
        fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in sorted(self.jobs.values(), key=lambda job: job['created_at']):
            if job['status'] not in TERMINAL:
                priority = float('-inf') if job.get('remote_started') else job['created_at']
                self.pending.put((priority, job['id']))
        self.thread = threading.Thread(target=self.run, name='studio-gpu-queue', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        self.file_lock.close()

    def update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(changes, updated_at=time.time())
            atomic_json(self.settings.data / job_id / 'job.json', self.jobs[job_id])
            return dict(self.jobs[job_id])

    def get(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise HTTPException(404, 'Asset not found')
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, 'Asset not found')
            return dict(self.jobs[job_id])

    def add(self, name, seed, images, *, preserve_face=False, face_reference=None):
        with self.lock:
            if sum(job['status'] not in TERMINAL for job in self.jobs.values()) >= 6:
                raise HTTPException(429, 'The queue is full. Wait for an asset to finish before adding another.')
            job_id = uuid.uuid4().hex
            folder = self.settings.data / job_id
            (folder / 'input').mkdir(parents=True)
            for i, data in enumerate(images, 1):
                (folder / 'input' / f'{i:03d}.png').write_bytes(data)
            if preserve_face:
                (folder / 'face-reference.png').write_bytes(face_reference or images[0])
            self.jobs[job_id] = {'id': job_id, 'name': name, 'seed': seed, 'status': 'queued',
                'stage': None, 'completed_stages': [], 'created_at': time.time(), 'cancel_requested': False,
                'message': 'In the generation queue', 'engine': self.engine,
                'encoder': getattr(self.bridge, 'encoder',
                    'Original DINOv3' if self.engine == 'mlx' else 'SoftEq DINOv3'),
                'view_count': len(images),
                'preserve_face': preserve_face,
                'face_reference_url': f'/api/jobs/{job_id}/face-reference' if preserve_face else None,
                'model_url': None, 'cover_url': f'/api/jobs/{job_id}/images/1'}
            job = self.update(job_id)
            self.pending.put((job['created_at'], job_id))
            return job

    def health(self):
        with self.health_lock:
            if time.monotonic() - self.health_cache[0] > 15:
                try:
                    value = self.bridge.health()
                except Exception:
                    value = {'connected': False, 'ready': False, 'local_mlx_ready': False,
                             'message': 'Generation engine unavailable. Check the Studio server log.'}
                self.health_cache = (time.monotonic(), value)
            return self.health_cache[1]

    def bridge_for(self, job):
        if isinstance(self.bridge, LocalBridge) and job.get('engine', 'cuda') == 'cuda':
            return RemoteBridge(self.settings)
        if isinstance(self.bridge, RemoteBridge) and job.get('engine') == 'mlx':
            return LocalBridge(self.settings)
        return self.bridge

    def run(self):
        while not self.stop.is_set():
            try:
                _, job_id = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.execute(job_id)
            except Exception as exc:
                # A detached remote job may still be running. Request cancellation of
                # this job only; never stop a shared GPU process or another service.
                if self.jobs[job_id].get('remote_started'):
                    try:
                        self.bridge_for(self.jobs[job_id]).cancel(job_id)
                    except Exception:
                        pass
                self.update(job_id, status='failed', message='Generation could not finish. Open details to retry.',
                            error=str(exc)[-4000:])
            finally:
                self.pending.task_done()

    def execute(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in TERMINAL:
                return
            folder = self.settings.data / job_id
            bridge = self.bridge_for(job)
            launch = not job.get('remote_started')
            if launch:
                if job['cancel_requested']:
                    self.update(job_id, status='cancelled', message='Generation cancelled')
                    return
                self.update(job_id, status='uploading', message='Preparing local generation' if job.get('engine') == 'mlx' else 'Sending views to your private GPU')
        if launch:
            bridge.start(job_id, folder, job['seed'])
            self.update(job_id, remote_started=True)
        failures = 0
        while not self.stop.is_set():
            try:
                if self.get(job_id)['cancel_requested']:
                    bridge.cancel(job_id)
                state = bridge.read_status(job_id)
                failures = 0
            except (subprocess.SubprocessError, OSError, ValueError):
                failures += 1
                self.update(job_id, message='Checking the local worker…' if job.get('engine') == 'mlx' else 'Reconnecting to the private GPU…')
                if failures >= 20:
                    raise RuntimeError('The generation worker could not be reached. Check Studio logs, then retry this asset.')
                self.stop.wait(3)
                continue
            selected = {key: state[key] for key in ['status', 'stage', 'completed_stages', 'message', 'error', 'seconds'] if key in state}
            if state['status'] == 'complete':
                self.update(job_id, status='downloading', message='Verifying your local asset' if job.get('engine') == 'mlx' else 'Bringing your asset back to this Mac',
                            completed_stages=state['completed_stages'], stage=None)
                metadata = bridge.fetch(job_id, folder, state['metadata'])
                if job.get('preserve_face'):
                    original = folder / 'generation.glb'
                    if not original.exists():
                        shutil.copyfile(folder / 'prediction.glb', original)
                    if glb_metadata(original) != metadata:
                        raise RuntimeError('The retained generation output does not match its receipt')
                    self.update(job_id, status='running', stage='face', message='Preserving the reference face')
                    try:
                        face = preserve_face(folder / 'prediction.glb', folder / 'face-reference.png',
                            folder / 'face-preservation',
                            cancelled=lambda: self.stop.is_set() or self.get(job_id)['cancel_requested'],
                            progress=lambda message: self.update(job_id, message=message))
                    except FaceCancelled:
                        if not self.stop.is_set():
                            self.update(job_id, status='cancelled', stage=None, message='Generation cancelled')
                        return
                    if face['status'] == 'applied':
                        candidate = folder / 'face-publication.download'
                        shutil.copyfile(face.pop('output'), candidate)
                        checked = glb_metadata(candidate)
                        if checked != face['metadata']:
                            candidate.unlink(missing_ok=True)
                            raise RuntimeError('Face-corrected GLB failed publication verification')
                        candidate.replace(folder / 'prediction.glb')
                        metadata = checked
                    receipt_path = folder / 'receipt.json'
                    generation_receipt = json.loads(receipt_path.read_text())
                    atomic_json(receipt_path, {'format': 'uniforge-studio-face-preservation-v1',
                        'engine': job['engine'], 'studio_job_id': job_id,
                        'generation_receipt': generation_receipt, 'face_preservation': face,
                        'output': metadata, **metadata})
                    self.update(job_id, face_preservation=face)
                    selected['message'] = 'Asset ready · face preserved' if face['status'] == 'applied' else face['message']
                self.update(job_id, **selected, metadata=metadata, model_url=f'/api/jobs/{job_id}/model',
                            download_url=f'/api/jobs/{job_id}/download', receipt_url=f'/api/jobs/{job_id}/receipt')
                return
            self.update(job_id, **selected)
            if state['status'] in TERMINAL:
                return
            self.stop.wait(3)


def create_app(settings=None, bridge=None):
    settings = settings or Settings()
    store = JobStore(settings, bridge or (LocalBridge(settings) if settings.engine == 'mlx' else RemoteBridge(settings)))

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        yield
        store.close()

    app = FastAPI(title='UniForge Studio', lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    allowed_origins = {f'http://{host}:{port}' for host in settings.allowed_hosts
                       for port in (settings.port, 5173)}

    @app.middleware('http')
    async def studio_boundary(request: Request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if request.headers.get('x-uniforge-studio') != '1' or (origin and origin not in allowed_origins):
                return JSONResponse({'detail': 'Open Studio at an allowed host to make changes.'}, status_code=403)
            length = request.headers.get('content-length', '0')
            if not length.isdigit() or int(length) > 64 * 1024**2:
                return JSONResponse({'detail': 'The upload must be smaller than 64 MB.'}, status_code=413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    @app.get('/api/health')
    def health():
        payload = dict(store.health())
        payload['face_preservation'] = face_readiness()
        return payload

    @app.get('/api/jobs')
    def jobs():
        with store.lock:
            return sorted([dict(job) for job in store.jobs.values()], key=lambda job: job['created_at'], reverse=True)

    @app.post('/api/jobs', status_code=202)
    async def create_job(files: list[UploadFile] = File(...), name: str = Form('Untitled asset'), seed: int = Form(42),
                         preserve_face: bool = Form(False),
                         face_reference: UploadFile | None = File(None)):
        try:
            if preserve_face and not face_readiness()['available']:
                raise HTTPException(422, 'Install the local face-correction worker before selecting Preserve face.')
            if face_reference is not None and not preserve_face:
                raise HTTPException(422, 'Select Preserve face to use a face reference.')
            min_views = getattr(store.bridge, 'min_views', 1 if store.engine == 'mlx' else 4)
            if not (min_views <= len(files) <= 4):
                raise HTTPException(422, 'Choose one to four views of the same object.' if min_views == 1 else 'Choose exactly four views of the same object.')
            if not 0 <= seed <= 2**32 - 1:
                raise HTTPException(422, 'Seed must be between 0 and 4294967295.')
            name = name.strip()
            if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
                raise HTTPException(422, 'Choose an asset name between 1 and 80 characters.')
            images = []
            for upload in files:
                try:
                    images.append(await run_in_threadpool(normalize_image, await upload.read(MAX_FILE + 1)))
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            reference = None
            if face_reference is not None:
                try:
                    reference = await run_in_threadpool(normalize_image, await face_reference.read(MAX_FILE + 1))
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            health = await run_in_threadpool(store.health)
            if not health.get('ready'):
                raise HTTPException(
                    503,
                    health.get('message') or 'The private GPU is not ready. Check its status and try again.',
                    headers={'Retry-After': '15'},
                )
            return store.add(name, seed, images,
                             preserve_face=preserve_face, face_reference=reference)
        finally:
            for upload in files:
                await upload.close()
            if face_reference is not None:
                await face_reference.close()

    @app.get('/api/jobs/{job_id}')
    def get_job(job_id: str):
        return store.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        with store.lock:
            job = store.get(job_id)
            if job['status'] in TERMINAL:
                return job
            if job['status'] == 'queued' and not job.get('remote_started'):
                return store.update(job_id, cancel_requested=True, status='cancelled', message='Generation cancelled')
            return store.update(job_id, cancel_requested=True, message='Cancelling after the current operation…')

    @app.post('/api/jobs/{job_id}/resume', status_code=202)
    def resume(job_id: str):
        with store.lock:
            job = store.get(job_id)
            bridge = store.bridge_for(job)
            if job['status'] != 'failed' or job.get('engine') != 'cuda3060' or not hasattr(bridge, 'prepare_resume'):
                raise HTTPException(409, 'Only a failed RTX 3060 job can resume its saved checkpoints.')
            if sum(item['status'] not in TERMINAL for item in store.jobs.values()) >= 6:
                raise HTTPException(429, 'The queue is full. Wait for an asset to finish before resuming.')
            try:
                bridge.prepare_resume(job_id)
            except (subprocess.SubprocessError, OSError) as exc:
                raise HTTPException(409, 'The saved worker is still active or cannot be resumed safely.') from exc
            saved = store.update(job_id, status='queued', stage=None, remote_started=False,
                                 cancel_requested=False, last_error=job.get('error'), error=None,
                                 resume_count=job.get('resume_count', 0) + 1,
                                 message='Resuming saved checkpoints after the current asset')
            store.pending.put((job['created_at'], job_id))
            return saved

    @app.get('/api/jobs/{job_id}/images/{index}')
    def job_image(job_id: str, index: int, thumbnail: bool = False):
        store.get(job_id)
        if index not in range(1, 5):
            raise HTTPException(404, 'Image not found')
        return image_response(settings.data / job_id / 'input' / f'{index:03d}.png', thumbnail)

    @app.get('/api/jobs/{job_id}/{kind}')
    def job_file(job_id: str, kind: str):
        job = store.get(job_id)
        if kind == 'face-reference' and job.get('preserve_face'):
            return image_response(settings.data / job_id / 'face-reference.png', False)
        if kind not in {'model', 'download', 'receipt'} or job['status'] != 'complete':
            raise HTTPException(404, 'This asset is not ready yet')
        if kind == 'receipt':
            return FileResponse(settings.data / job_id / 'receipt.json', media_type='application/json', filename='generation-receipt.json')
        filename = re.sub(r'[^a-zA-Z0-9_-]+', '-', job['name']).strip('-')[:80] or 'asset'
        return FileResponse(settings.data / job_id / 'prediction.glb', media_type='model/gltf-binary',
                            filename=f'{filename}.glb' if kind == 'download' else None)

    @app.get('/api/example')
    def example():
        if store.engine == 'mlx':
            model = ROOT / 'run_logs/mlx-case02-seed42/model.glb'
            return {'name': 'Golden figure', 'images': ['/api/example/images/1'],
                    'model_url': '/api/example/model' if model.exists() else None,
                    'download_url': '/api/example/download' if model.exists() else None,
                    'cover_url': '/api/example/images/1', 'id': 'example', 'status': 'complete',
                    'encoder': 'Original DINOv3', 'engine': 'mlx', 'view_count': 1,
                    'source': 'Verified local MLX example',
                    'metadata': glb_metadata(model) if model.exists() else None}
        model = ROOT / 'outputs/cuda-20261020/case01/prediction.glb'
        return {'name': 'Monkey King', 'images': [f'/api/example/images/{i}' for i in range(1, 5)],
                'model_url': '/api/example/model' if model.exists() else None,
                'download_url': '/api/example/download' if model.exists() else None,
                'cover_url': '/api/example/images/1', 'id': 'example', 'status': 'complete',
                'encoder': 'Original DINOv3', 'source': 'Previously generated example',
                'metadata': glb_metadata(model) if model.exists() else None}

    @app.get('/api/example/images/{index}')
    def example_image(index: int, thumbnail: bool = False):
        if index not in range(1, 5):
            raise HTTPException(404, 'Image not found')
        case = 'case02' if store.engine == 'mlx' else 'case01'
        return image_response(ROOT / 'assets' / case / f'{index:03d}.png', thumbnail)

    @app.get('/api/example/{kind}')
    def example_model(kind: str):
        model = ROOT / ('run_logs/mlx-case02-seed42/model.glb' if store.engine == 'mlx' else 'outputs/cuda-20261020/case01/prediction.glb')
        if kind not in {'model', 'download'} or not model.exists():
            raise HTTPException(404, 'Example model is unavailable')
        filename = 'Golden-figure-MLX.glb' if store.engine == 'mlx' else 'Monkey-King-example.glb'
        return FileResponse(model, media_type='model/gltf-binary', filename=filename if kind == 'download' else None)

    dist = ROOT / 'studio/web/dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='studio')
    return app


app = create_app()
