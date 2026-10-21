"""Private-host worker. Reuses the tested staged pipeline without changing it.

Only this controller's child process group is cancelled. A filesystem GPU lock
serializes Studio workers; an occupied GPU is left alone and polled until free.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

HOME = Path('/home/op/uniforge3d-work')
REPO = HOME / 'UniForge3D'
ENCODER_SHA = '531d79166c0fb21fb2aafc6d7296ff9f881f96d9b9934ab87cf367b13833fa12'
GPU = 'GPU-828df6fd-3fd0-ed25-0b2b-2b6d9d8dca47'
PHASES = ['preprocess', 'dino', 'structure', 'shape512', 'shape1024', 'texture', 'decode']


def sha256_file(path):
    """Streaming SHA256, including the CUDA environment's Python 3.10."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def gpu_free():
    result = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used = result.stdout.strip()
    if not used.isdigit():
        raise RuntimeError('The private host NVIDIA GPU is unavailable. Restore GPU access before retrying.')
    return int(used) < 100


def glb_metadata(path):
    with path.open('rb') as stream:
        magic, version, length = struct.unpack('<4sII', stream.read(12))
        if magic != b'glTF' or version != 2 or length != path.stat().st_size:
            raise ValueError('The generator did not produce a complete GLB 2.0 file.')
        size, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or size > 16 * 1024 * 1024:
            raise ValueError('Invalid GLB metadata chunk.')
        model = json.loads(stream.read(size))
    triangles = vertices = 0
    for mesh in model.get('meshes', []):
        for primitive in mesh['primitives']:
            vertices += model['accessors'][primitive['attributes']['POSITION']]['count']
            if 'indices' in primitive:
                triangles += model['accessors'][primitive['indices']]['count'] // 3
    if vertices == 0 or triangles == 0:
        raise ValueError('The generated asset contains no mesh.')
    return {'bytes': length, 'sha256': sha256_file(path),
            'vertices': vertices, 'triangles': triangles, 'texture_size': 2048}


def run(job):
    job = job.resolve()
    # The caller may only launch folders inside this service's job directory.
    if job.parent != (HOME / 'studio/jobs').resolve() or len(job.name) != 32:
        raise ValueError('Invalid Studio job directory')
    int(job.name, 16)
    lock = (job / 'controller.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    status_path = job / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status') in {'complete', 'cancelled', 'failed'}:
        return
    request = json.loads((job / 'request.json').read_text())
    state = {'status': 'waiting', 'stage': None, 'completed_stages': [], 'started_at': time.time()}
    child = None

    def update(**changes):
        state.update(changes, updated_at=time.time())
        write_json(status_path, state)

    def cancelled():
        return (job / 'cancel').exists()

    try:
        update(message='Waiting for the private GPU')
        gpu_lock = (HOME / 'studio/gpu.lock').open('w')
        while True:
            if cancelled():
                update(status='cancelled', message='Generation cancelled')
                return
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if gpu_free():
                    break
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            time.sleep(3)
        if __import__('shutil').disk_usage(job).free < 4 * 1024**3:
            raise RuntimeError('The private GPU host needs at least 4 GB of free disk space.')
        encoder = HOME / 'studio/encoder'
        if sha256_file(encoder / 'model.safetensors') != ENCODER_SHA:
            raise RuntimeError('SoftEq encoder verification failed; restore the verified encoder.')
        sys.path.insert(0, str(REPO / 'tools'))
        from run_cuda_staged import make_config
        config = make_config(HOME / 'weights')
        config['external_models']['dino'] = str(encoder)
        config['default_seed'] = int(request['seed'])
        inputs = {f'{i:03d}.png': sha256_file(job / 'input' / f'{i:03d}.png')
                  for i in range(1, 5)}
        if inputs != request['inputs']:
            raise RuntimeError('Reference image verification failed during transfer.')
        work = job / 'work'
        work.mkdir(exist_ok=True)
        write_json(work / 'config.json', config)
        write_json(work / 'inputs.json', inputs)
        env = dict(os.environ, CUDA_HOME='/usr/local/cuda-12.8', CUDA_VISIBLE_DEVICES=GPU,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ATTN_BACKEND='flash_attn',
                   SPARSE_ATTN_BACKEND='flash_attn', SPARSE_CONV_BACKEND='flex_gemm',
                   PYTHONUNBUFFERED='1')
        for phase in PHASES:
            if cancelled():
                update(status='cancelled', message='Generation cancelled')
                return
            done = work / f'{phase}.done.json'
            if not done.exists():
                update(status='running', stage=phase, message=f'Running {phase}')
                with (work / f'{phase}.log').open('w') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(REPO / 'tools/run_cuda_staged.py'),
                        '--input', str(job / 'input'), '--output', str(job), '--work', str(work), '--phase', phase],
                        stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env, start_new_session=True)
                    deadline = time.monotonic() + 1800
                    while child.poll() is None:
                        if cancelled() or time.monotonic() > deadline:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            if cancelled():
                                update(status='cancelled', message='Generation cancelled')
                                return
                            raise RuntimeError(f'The {phase} stage exceeded its 30 minute limit.')
                        if time.time() - state['updated_at'] > 10:
                            update()
                        time.sleep(0.5)
                    if child.returncode:
                        tail = (work / f'{phase}.log').read_text(errors='replace')[-3000:]
                        raise RuntimeError(f'{phase} failed. {tail}')
                    child = None
            state['completed_stages'].append(json.loads(done.read_text()))
            update()
        metadata = glb_metadata(work / 'prediction.glb')
        receipt = {'engine': 'cuda', 'precision': 'source', 'encoder': 'SoftEq DINOv3 FP16 storage / FP32 compute',
                   'encoder_sha256': ENCODER_SHA, 'inputs': inputs, 'seed': request['seed'], **metadata}
        write_json(job / 'receipt.json', receipt)
        update(status='complete', stage=None, message='Asset ready', metadata=metadata,
               finished_at=time.time(), seconds=time.time() - state['started_at'])
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        update(status='failed', message='Generation failed. See the job log for details.', error=str(exc)[-4000:])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    run(parser.parse_args().job)
