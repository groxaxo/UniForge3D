"""Optional CPU face-photo correction; the original GLB remains untouched."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .remote_runner import glb_metadata

ROOT = Path(__file__).resolve().parents[2]
FACE_TOOL = ROOT / 'tools/ComfyUI-GLB-Face-Retexture'


class FaceCancelled(Exception):
    """The user cancelled, or the API is restarting; never call this a match failure."""


def face_readiness():
    available = ((FACE_TOOL / '.venv/bin/python').is_file()
                 and all((FACE_TOOL/'models'/name).is_file() for name in
                         ('face_landmarker.task', 'selfie_multiclass.tflite'))
                 and (ROOT/'tools/glb_face_compat.py').is_file())
    return {'available': available,
            'message': 'Restore a visible face from a matching reference photo.' if available
                       else 'The local face-correction worker needs to be installed.'}


def run_process(command, log, cancelled):
    with Path(log).open('ab') as handle:
        process = subprocess.Popen([str(value) for value in command], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            while process.poll() is None:
                if cancelled():
                    raise FaceCancelled()
                if time.monotonic() > deadline:
                    raise RuntimeError('Face correction timed out')
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError('Face worker refused the reference or source; see face-worker.log')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def preserve_face(source, reference, workspace, cancelled=lambda: False, progress=lambda message: None):
    """Return a verified corrected candidate, or an explicit skipped result.

    Source publication and receipt updates belong to the queue owner. A failed
    detection, unsupported asset or failed preservation gate keeps the original.
    Each attempt is retained separately for restart/debugging; no partial candidate
    is treated as a successfully preserved face.
    """
    source, reference, workspace = map(Path, (source, reference, workspace))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    base = {'requested': True, 'source_sha256': source_hash}
    attempt = workspace / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    try:
        if not face_readiness()['available']:
            raise RuntimeError('Face worker unavailable')
        python = FACE_TOOL/'.venv/bin/python'
        progress('Preparing the mesh for face preservation')
        run_process([python, ROOT/'tools/glb_face_compat.py', source, attempt/'compatible.glb',
                     '--proof', attempt/'compatibility.json'], attempt/'face-worker.log', cancelled)
        progress('Matching the reference face and preserving its texture')
        run_process([python, FACE_TOOL/'worker.py', '--source-glb', attempt/'compatible.glb',
            '--reference-image', reference, '--output-glb', attempt/'corrected.glb',
            '--comparison-png', attempt/'comparison.png', '--proof-json', attempt/'proof.json',
            '--model-dir', FACE_TOOL/'models', '--strength', '1', '--feather', '.18',
            '--light-balance', '1', '--tone-match', '1', '--skin-recolor', '0'],
            attempt/'face-worker.log', cancelled)
        proof = json.loads((attempt/'proof.json').read_text())
        if not (proof.get('geometry_unchanged') is True
                and proof.get('original_embedded_images_exact') is True
                and proof.get('protected_body_changed_render_pixels') == 0
                and proof.get('original_triangle_count', 0) > 0
                and proof.get('output_triangle_count') == proof['original_triangle_count']
                and proof.get('changed_head_render_pixels', 0) > 0):
            raise RuntimeError('Face preservation verification failed')
        metadata = glb_metadata(attempt/'corrected.glb')
        if metadata['triangles'] != proof['output_triangle_count']:
            raise RuntimeError('Corrected geometry does not match its preservation proof')
        if cancelled():
            raise FaceCancelled()
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('The source GLB changed during face correction')
        return {**base, 'status': 'applied', 'message': 'Face preserved from your reference photo.',
                'output': str(attempt/'corrected.glb'), 'metadata': metadata,
                'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                'proof': proof, 'attempt': attempt.name}
    except FaceCancelled:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        (attempt/'failure.txt').write_text(f'{type(error).__name__}: {error}\n')
        return {**base, 'status': 'skipped', 'attempt': attempt.name,
                'message': 'Face correction could not be applied. Original face texture kept.'}
