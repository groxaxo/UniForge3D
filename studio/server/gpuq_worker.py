"""Foreground .51 CUDA worker; it may run only after one-GPU GPUQ admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from .gpuq_common import PHASES, atomic_json, glb_metadata, job_folder, sha256_file

HOME = Path('/home/op/uniforge3d-work')


def admission(environment):
    identifier = environment.get('GPU_QUEUE_JOB_ID', '')
    visibility = environment.get('CUDA_VISIBLE_DEVICES', '')
    if (not re.fullmatch(r'[0-9a-f]{12}', identifier) or not visibility
            or len(visibility.split(',')) != 1 or not environment.get('GPU_QUEUE_LOCK_FDS')):
        raise RuntimeError('UniForge requires an admitted one-GPU GPUQ job')
    return identifier, visibility


def run(folder, home=HOME):
    # Fail before importing CUDA, even if somebody invokes this module directly.
    queue_id, visible = admission(os.environ)
    home = Path(home)
    folder = Path(folder).resolve()
    if folder != job_folder(home / 'studio-gpuq/jobs', folder.name).resolve():
        raise ValueError('Invalid GPUQ workspace')
    repo = home / 'UniForge3D'
    runtime = (home / 'studio-gpuq/runtime').resolve()
    status = folder / 'status.json'
    state = dict(status='running', stage=None, completed_stages=[], started_at=time.time(),
                 queue_id=queue_id, gpu_visibility=visible)
    child = None

    def update(**values):
        state.update(values, updated_at=time.time())
        atomic_json(status, state)

    def interrupted(signum, frame):
        raise RuntimeError('GPUQ stopped the admitted worker; checkpoints were retained')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if (folder / 'cancel').exists():
            raise RuntimeError('Generation cancelled before the worker started')
        request = json.loads((folder / 'request.json').read_text())
        manifest = json.loads((runtime / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha256_file(runtime / path) != digest:
                raise RuntimeError('Pinned CUDA runtime verification failed')
        sys.path.insert(0, str(runtime / 'tools'))
        from run_cuda_staged import make_config, input_images
        images = input_images(folder / 'input')
        inputs = {path.name: sha256_file(path) for path in images}
        if inputs != request['inputs']:
            raise RuntimeError('Reference image verification failed')
        verification = json.loads((home / 'weights/dino-hf/verification.json').read_text())
        if not verification.get('verified') or not all(t['cosine'] >= .999 for t in verification['tests']):
            raise RuntimeError('Original DINOv3 encoder is not verified')
        config = make_config(home / 'weights')
        config['default_seed'] = int(request['seed'])
        work = folder / 'work'
        work.mkdir(exist_ok=True)
        atomic_json(work / 'config.json', config)
        atomic_json(work / 'inputs.json', inputs)
        # Preserve GPUQ's CUDA_VISIBLE_DEVICES, ownership and inherited process group.
        env = dict(os.environ, CUDA_HOME='/usr/local/cuda-12.8', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', ATTN_BACKEND='flash_attn',
                   SPARSE_ATTN_BACKEND='flash_attn', SPARSE_CONV_BACKEND='flex_gemm',
                   PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   TORCH_EXTENSIONS_DIR=str(home / 'native-cache'))
        env['PATH'] = '/usr/local/cuda-12.8/bin:' + env.get('PATH', '')
        gpu = subprocess.run(['nvidia-smi', '-i', visible, '--query-gpu=uuid,name,memory.total',
                              '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()
        for phase in PHASES:
            if (folder / 'cancel').exists():
                raise RuntimeError('Generation cancelled')
            update(stage=phase, message=f'RTX 3090 · GPUQ · {phase}')
            done = work / f'{phase}.done.json'
            if not done.exists():
                with (work / f'{phase}.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(runtime / 'tools/run_cuda_staged.py'),
                        '--input', str(folder / 'input'), '--output', str(folder), '--work', str(work),
                        '--phase', phase], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 1800
                    while child.poll() is None:
                        if (folder / 'cancel').exists() or time.monotonic() > deadline:
                            raise RuntimeError(f'Generation cancelled or {phase} exceeded 30 minutes')
                        if time.time() - state['updated_at'] > 10:
                            update()
                        time.sleep(.5)
                    if child.returncode:
                        raise RuntimeError(f'{phase} failed: ' + (work / f'{phase}.log').read_text(errors='replace')[-3000:])
                    child = None
            state['completed_stages'].append(json.loads(done.read_text()))
            update()
        metadata = glb_metadata(work / 'prediction.glb')
        atomic_json(folder / 'receipt.json', dict(engine='cuda3090', hostname=socket.gethostname(),
            queue='gpuq', queue_id=queue_id, gpu=gpu, cuda_visible_devices=visible,
            runtime_manifest=manifest,
            physical_gpus=os.environ.get('GPU_QUEUE_PHYSICAL_GPUS'),
            precision='CUDA source precision', encoder='Original DINOv3',
            encoder_sha256=sha256_file(home / 'weights/dino-hf/model.safetensors'),
            worker_sha256=sha256_file(Path(__file__)), seed=request['seed'], inputs=inputs,
            stages=state['completed_stages'], **metadata))
        update(status='complete', stage=None, message='Asset ready · RTX 3090 · GPUQ',
               metadata=metadata, seconds=time.time() - state['started_at'])
    except BaseException as error:
        # Children share the GPUQ-owned process group; never daemonize a stage.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        update(status='failed', error=str(error)[-4000:], message='GPUQ generation stopped; checkpoints retained')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    run(parser.parse_args().job)
