"""Separate Studio queue for the private RTX 3060; the MLX service stays independent."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from .app import ROOT, Settings, RemoteBridge, create_app

GPU = 'GPU-72f3cda5-5f26-bed3-6093-0608adb365e3'


class Cuda3060Bridge(RemoteBridge):
    engine = 'cuda3060'
    encoder = 'Original DINOv3'
    min_views = 1

    def health(self):
        code = f'''import json, shutil, subprocess
from pathlib import Path
p = Path({self.settings.remote!r})
v = Path('/home/op/uniforge3d-work/weights/dino-hf/verification.json')
r = subprocess.run(['nvidia-smi', '-i', {GPU!r}, '--query-gpu=name,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
g = r.stdout.strip().split(',')
encoder = v.exists() and json.loads(v.read_text()).get('verified', False)
valid = r.returncode == 0 and len(g) == 3 and all(x.strip().isdigit() for x in g[1:])
disk = shutil.disk_usage(p).free
cpu_mesh = Path('/home/op/uniforge3d-work/cpu-mesh-deps/fast_simplification/__init__.py').exists()
ready = valid and encoder and cpu_mesh and (p/'cuda3060_runner.py').exists() and disk > 2*1024**3
print(json.dumps(dict(connected=True, ready=ready, gpu=g[0].strip() if valid else 'RTX 3060 unavailable',
    used_mib=int(g[1]) if valid else 0, total_mib=int(g[2]) if valid else 0, free_bytes=disk,
    encoder_ready=encoder, cpu_mesh_ready=cpu_mesh, worker_ready=(p/'cuda3060_runner.py').exists(),
    message='RTX 3060 on .54; one job at a time. Existing embedding service is preserved.' if ready
            else 'RTX 3060 setup needs its verified encoder, worker, CPU mesh helper and 2 GB free.')))
'''
        result = self.ssh(f'{shlex.quote(self.settings.python)} -c {shlex.quote(code)}',
                          timeout=30, capture_output=True, text=True)
        value = json.loads(result.stdout)
        value.update(engine='CUDA · RTX 3060 on .54 · original encoder', engine_id=self.engine,
                     encoder=self.encoder, min_views=1, max_views=4, local_mlx_ready=False)
        return value

    def start(self, job_id, directory, seed):
        remote_job = f'{self.settings.remote}/jobs/{job_id}'
        self.ssh(f'mkdir -p {shlex.quote(remote_job + "/input")}')
        paths = sorted((directory / 'input').glob('*.png'))
        for path in paths:
            with path.open('rb') as stream:
                self.ssh(f'cat > {shlex.quote(remote_job + "/input/" + path.name)}', stdin=stream,
                         timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        payload = json.dumps({'seed': seed, 'inputs': {
            path.name: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for path in paths}}).encode()
        self.ssh(f'cat > {shlex.quote(remote_job + "/request.json")}', input=payload, capture_output=True)
        command = (f'nohup {shlex.quote(self.settings.python)} -u '
                   f'{shlex.quote(self.settings.remote + "/cuda3060_runner.py")} --job {shlex.quote(remote_job)} '
                   f'> {shlex.quote(remote_job + "/controller.log")} 2>&1 < /dev/null &')
        self.ssh(command, capture_output=True)

    def prepare_resume(self, job_id):
        remote_job = f'{self.settings.remote}/jobs/{job_id}'
        code = f'''import fcntl, json, time
from pathlib import Path
p = Path({remote_job!r})
with (p/'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((p/'status.json').read_text())
    if state['status'] != 'failed' or (p/'cancel').exists():
        raise RuntimeError('Worker is not a failed, uncancelled job')
    stamp = str(time.time_ns())
    for path in [p/'status.json', p/'controller.log', p/'work/decode.log']:
        if path.exists():
            path.rename(path.with_name(path.name+'.failed.'+stamp))
'''
        self.ssh(f'{shlex.quote(self.settings.python)} -c {shlex.quote(code)}',
                 timeout=20, capture_output=True, text=True)


settings = Settings(data=ROOT / 'outputs/studio-3060', host='op@100.85.200.54',
                    remote='/home/op/uniforge3d-work/studio-3060', engine='cuda3060', port=7863)
app = create_app(settings, Cuda3060Bridge(settings))
