"""Detached per-job MLX worker, run with the repository's model environment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

from .remote_runner import glb_metadata, write_json

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'preprocess': 'Prepare images', 'dino': 'Read visual features',
          'structure': 'Build structure', 'shape512': 'Shape the object',
          'shape1024': 'Refine geometry', 'texture': 'Generate texture', 'decode': 'Bake & export'}


def run(folder):
    import mlx.core as mx
    from uniforge3d_mlx.pipeline import Pipeline

    folder = folder.resolve()
    if folder.parent != (ROOT / 'outputs/studio').resolve() or not re.fullmatch('[0-9a-f]{32}', folder.name):
        raise ValueError('Invalid local Studio job directory')
    job = json.loads((folder / 'job.json').read_text())
    if job['engine'] != 'mlx':
        raise ValueError('This worker only runs local MLX jobs')
    state = {'status': 'waiting', 'stage': None, 'completed_stages': [], 'started_at': time.time()}

    def update(**changes):
        state.update(changes, updated_at=time.time())
        write_json(folder / 'status.json', state)

    done = threading.Event()

    def watch_cancel():
        while not done.wait(0.5):
            if (folder / 'cancel').exists():
                # Terminate only this dedicated worker; release Metal allocations
                # even when a native UV operation is holding the Python GIL.
                os.kill(os.getpid(), 15)

    threading.Thread(target=watch_cancel, daemon=True).start()
    lock = (folder.parent / 'mlx-gpu.lock').open('w')
    try:
        update(message='Waiting for local MLX generation')
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                done.wait(1)
        mx.set_cache_limit(1024**3)
        mx.set_memory_limit(mx.device_info()['max_recommended_working_set_size'])
        elapsed = {}

        def log(message):
            print(message, flush=True)
            step = re.match(r'(\w+): step (\d+)/(\d+)', message)
            if step:
                phase, current, total = step.groups()
                update(message=f'{LABELS.get(phase, phase)} · step {current} of {total}')

        class StudioPipeline(Pipeline):
            def stage(self, name, fn):
                phase = {'handoff': 'shape512', 'export': 'decode'}.get(name, name)
                update(status='running', stage=phase, message=LABELS[phase])
                start = time.monotonic()
                result = super().stage(name, fn)
                elapsed[phase] = elapsed.get(phase, 0) + time.monotonic() - start
                finish = {'shape512': 'handoff', 'decode': 'export'}.get(phase, phase)
                if name == finish:
                    state['completed_stages'].append({'phase': phase, 'seconds': elapsed[phase]})
                    update()
                return result

        images = sorted((folder / 'input').glob('*.png'))
        pipeline = StudioPipeline(ROOT / 'weights/UniForge3D-oQ5-complete', folder / 'work',
                                  seed=job['seed'], resume=(folder / 'work/run.json').exists(), log=log)
        output = pipeline.run(images)
        metadata = glb_metadata(output)
        receipt = json.loads((folder / 'work/run.json').read_text())
        receipt.update(engine='mlx', encoder='Original DINOv3', studio_job_id=job['id'],
                       inputs_sha256={p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in images},
                       output=metadata)
        write_json(folder / 'receipt.json', receipt)
        update(status='complete', stage=None, message='Asset ready', metadata=metadata,
               seconds=time.time() - state['started_at'])
    except Exception as exc:
        traceback.print_exc()
        update(status='failed', stage=None, message='Local generation could not finish. Open details to retry.',
               error=f'{type(exc).__name__}: {exc}')
    finally:
        done.set()
        mx.synchronize()
        mx.clear_cache()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--token', required=True)
    run(parser.parse_args().job)
