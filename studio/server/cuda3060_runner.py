"""RTX 3060 worker using read-only shared source weights and isolated stage processes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

HOME = Path('/home/op/uniforge3d-work')
REPO = HOME / 'UniForge3D'
BASE = HOME / 'studio-3060'
GPU = 'GPU-72f3cda5-5f26-bed3-6093-0608adb365e3'
PHASES = ['preprocess', 'dino', 'structure', 'shape512', 'shape1024', 'texture', 'decode']
sys.path.insert(0, str(REPO))
from studio.server.remote_runner import write_json, glb_metadata, sha256_file


def gpu_state():
    r = subprocess.run(['nvidia-smi', '-i', GPU, '--query-gpu=uuid,name,memory.total,memory.used',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    uuid, name, total, used = [x.strip() for x in r.stdout.strip().split(',')]
    if uuid != GPU or '3060' not in name:
        raise RuntimeError('The requested RTX 3060 was not found')
    return {'uuid': uuid, 'name': name, 'total_mib': int(total), 'used_mib': int(used)}


def run(job):
    job = job.resolve()
    if job.parent != (BASE / 'jobs').resolve() or len(job.name) != 32:
        raise ValueError('Invalid job directory')
    int(job.name, 16)
    lock = (job / 'controller.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    status = job / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') in {'complete', 'cancelled', 'failed'}:
        return
    state = {'status': 'waiting', 'stage': None, 'completed_stages': [], 'started_at': time.time()}
    child = None

    def update(**changes):
        state.update(changes, updated_at=time.time())
        write_json(status, state)

    def cancelled():
        return (job / 'cancel').exists()

    try:
        gpu_lock = (BASE / 'gpu.lock').open('w')
        update(message='Waiting for RTX 3060 headroom; existing services stay running')
        while True:
            if cancelled():
                update(status='cancelled', message='Generation cancelled')
                return
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu = gpu_state()
                # Allow the small existing embedding service, never evict another workload.
                if gpu['used_mib'] < 2048 and gpu['total_mib'] - gpu['used_mib'] >= 10240:
                    break
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            time.sleep(3)
        if shutil.disk_usage(job).free < 2 * 1024**3:
            raise RuntimeError('RTX 3060 scratch disk needs 2 GB free')
        request = json.loads((job / 'request.json').read_text())
        sys.path.insert(0, str(REPO / 'tools'))
        from run_cuda_staged import make_config, input_images
        images = input_images(job / 'input')
        inputs = {p.name: sha256_file(p) for p in images}
        if inputs != request['inputs']:
            raise RuntimeError('Reference image transfer hash mismatch')
        verification = json.loads((HOME / 'weights/dino-hf/verification.json').read_text())
        if not verification.get('verified') or not all(t['cosine'] >= .999 for t in verification['tests']):
            raise RuntimeError('Original DINOv3 encoder verification failed')
        encoder_sha = sha256_file(HOME / 'weights/dino-hf/model.safetensors')
        config = make_config(HOME / 'weights')
        config['default_seed'] = int(request['seed'])
        work = job / 'work'
        work.mkdir(exist_ok=True)
        write_json(work / 'config.json', config)
        write_json(work / 'inputs.json', inputs)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CUDA_HOME='/usr/local/cuda-12.8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ATTN_BACKEND='flash_attn',
                   SPARSE_ATTN_BACKEND='flash_attn', SPARSE_CONV_BACKEND='flex_gemm',
                   PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   TORCH_EXTENSIONS_DIR=str(HOME / 'native-cache'))
        env['PATH'] = '/usr/local/cuda-12.8/bin:' + env.get('PATH', '')
        for phase in PHASES:
            if cancelled():
                update(status='cancelled', message='Generation cancelled')
                return
            done = work / f'{phase}.done.json'
            if not done.exists():
                update(status='running', stage=phase, message=f'RTX 3060 · {phase}', gpu=gpu)
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
                            raise RuntimeError(f'{phase} exceeded 30 minutes')
                        if time.time() - state['updated_at'] > 10:
                            update(message=f'RTX 3060 · {phase} · {int(time.time()-state["started_at"])}s elapsed')
                        time.sleep(.5)
                    if child.returncode:
                        raise RuntimeError(f'{phase} failed: ' + (work / f'{phase}.log').read_text(errors='replace')[-4000:])
                    child = None
            state['completed_stages'].append(json.loads(done.read_text()))
            update()
        metadata = glb_metadata(work / 'prediction.glb')
        receipt = dict(engine='cuda3060', gpu=gpu_state(), hostname=socket.gethostname(),
                       weights_location='source from private .51 via read-only SSHFS; verified shared-backbone cache and CUDA compute on .54',
                       precision='CUDA source precision', encoder='Original DINOv3', encoder_sha256=encoder_sha,
                       seed=request['seed'], inputs=inputs, stages=state['completed_stages'], **metadata)
        write_json(job / 'receipt.json', receipt)
        update(status='complete', stage=None, message='Asset ready · RTX 3060', metadata=metadata,
               finished_at=time.time(), seconds=time.time()-state['started_at'])
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=15)
        update(status='failed', message='RTX 3060 generation failed; see details', error=str(exc)[-4000:])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    run(parser.parse_args().job)
