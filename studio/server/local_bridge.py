"""Studio's local MLX process bridge; neural dependencies stay in the model venv."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time
import uuid

from .remote_runner import glb_metadata, write_json

ROOT = Path(__file__).resolve().parents[2]
EXPORT_LABELS = {
    'load_npz': 'Reading generated geometry and materials',
    'extract_mesh': 'Extracting the mesh',
    'remesh': 'Refining the surface',
    'simplify_mesh': 'Optimizing the mesh',
    'uv': 'Building the texture atlas',
    'texture_bake': 'Baking color and material maps',
    'write_glb': 'Writing the GLB',
}


class LocalBridge:
    engine = 'mlx'

    def __init__(self, settings):
        self.settings = settings

    def folder(self, job_id):
        return self.settings.data / job_id

    def health(self):
        python = ROOT / '.venv/bin/python'
        bundle = ROOT / 'weights/UniForge3D-oQ5-complete'
        result = {'connected': True, 'ready': False, 'local_mlx_ready': False,
                  'engine_id': 'mlx', 'engine': 'Local MLX · selective oQ5',
                  'encoder': 'Original DINOv3', 'min_views': 1, 'max_views': 4}
        try:
            if platform.system() != 'Darwin' or platform.machine() != 'arm64':
                raise ValueError('Local MLX generation requires an Apple Silicon Mac.')
            manifest = json.loads((bundle / 'bundle.json').read_text())
            if manifest.get('component_count') != 15 or not manifest.get('all_component_integrity_audits_passed'):
                raise ValueError('The complete, verified MLX model bundle is required.')
            for name, component in manifest['components'].items():
                for filename, info in component['files'].items():
                    if (bundle / name / filename).stat().st_size != info['bytes']:
                        raise ValueError(f'The MLX bundle needs repair: {name}/{filename}.')
            probe = subprocess.run([str(python), '-c',
                'import json; import mlx.core as mx; import mlx_spatial.spatialkit; '
                'print(json.dumps({"gpu":mx.device_info()["device_name"]}))'],
                cwd=ROOT, capture_output=True, text=True, check=True, timeout=15)
            result.update(json.loads(probe.stdout))
            result['free_bytes'] = shutil.disk_usage(self.settings.data).free
            if result['free_bytes'] < 4 * 1024**3:
                raise ValueError('Free at least 4 GB of disk space before generating an asset.')
            result.update(ready=True, local_mlx_ready=True,
                          message='Ready to generate on this Mac with the verified MLX bundle.')
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            result['message'] = f'Local MLX is unavailable: {str(exc)[:300]}'
        return result

    def alive(self, job_id):
        try:
            control = json.loads((self.folder(job_id) / 'controller.json').read_text())
            command = subprocess.run(['ps', '-p', str(int(control['pid'])), '-o', 'command='],
                                     capture_output=True, text=True, timeout=5).stdout
            # A persisted PID alone is not authority to signal a process after restart.
            return all(value in command for value in (
                'studio.server.local_runner', str(self.folder(job_id).resolve()), control['token']))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return False

    def start(self, job_id, directory, seed):
        if self.alive(job_id):
            return
        token = uuid.uuid4().hex
        write_json(directory / 'status.json', {'status': 'waiting', 'stage': None,
                   'completed_stages': [], 'message': 'Preparing local MLX generation'})
        with (directory / 'worker.log').open('ab') as log:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m',
                'studio.server.local_runner', '--job', str(directory.resolve()), '--token', token],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, 'MLX_SPATIAL_STAGE_EVENTS_PATH': str(directory.resolve() / 'export-events.jsonl')})
        write_json(directory / 'controller.json', {'pid': process.pid, 'token': token})

    def read_status(self, job_id):
        folder = self.folder(job_id)
        state = json.loads((folder / 'status.json').read_text())
        if state['status'] not in {'complete', 'failed', 'cancelled'} and not self.alive(job_id):
            cancelled = (folder / 'cancel').exists()
            state.update(status='cancelled' if cancelled else 'failed', stage=None,
                         message='Generation cancelled' if cancelled else 'The local worker stopped. Retry this asset.')
            if not cancelled:
                state['error'] = (folder / 'worker.log').read_text(errors='replace')[-4000:]
            write_json(folder / 'status.json', state)
        if state['status'] == 'running' and state.get('stage') == 'decode':
            self.export_progress(folder, state)
        return state

    @staticmethod
    def export_progress(folder, state):
        """Read native events outside the worker, even while native code holds its GIL."""
        try:
            data = (folder / 'export-events.jsonl').read_text()
            # The writer may be between write and newline; use only complete records.
            lines = data.split('\n')[:-1]
            if not lines:
                return
            active = {}
            for line in lines:
                event = json.loads(line)
                # Native UV work emits nested uv.* events. Keep the active parent
                # operation so the UI retains its label and total elapsed time.
                stage = event['stage']
                if '.' in stage:
                    continue
                if event['event'] == 'start':
                    active[stage] = event
                else:
                    active.pop(stage, None)
            if not active:
                return
            event = next(reversed(active.values()))
            label = EXPORT_LABELS.get(event['stage'], 'Checking mesh quality')
            seconds = max(0, int(time.monotonic() - float(event['monotonic_sec'])))
            state['message'] = f'{label} · {seconds // 60}m {seconds % 60:02d}s elapsed'
            if event['stage'] == 'uv':
                state['message'] += '. This step can take several minutes.'
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            # Missing/partially written diagnostics must never fail a valid generation.
            pass

    def cancel(self, job_id):
        folder = self.folder(job_id)
        (folder / 'cancel').touch()
        if self.alive(job_id):
            control = json.loads((folder / 'controller.json').read_text())
            try:
                os.killpg(int(control['pid']), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def fetch(self, job_id, directory, metadata):
        source = directory / 'work/model.glb'
        checked = glb_metadata(source)
        if checked['sha256'] != metadata['sha256'] or checked['bytes'] != metadata['bytes']:
            raise RuntimeError('The generated GLB changed before it could be published.')
        temporary = directory / 'prediction.download'
        shutil.copyfile(source, temporary)
        if glb_metadata(temporary) != checked:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('The local GLB copy failed verification.')
        temporary.replace(directory / 'prediction.glb')
        return checked
