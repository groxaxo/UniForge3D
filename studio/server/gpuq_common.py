"""CPU-only protocol helpers shared by the .51 gateway and admitted worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

PHASES = ('preprocess', 'dino', 'structure', 'shape512', 'shape1024', 'texture', 'decode')
QUEUE_TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def job_folder(base, identifier):
    if not re.fullmatch(r'[0-9a-f]{32}', identifier):
        raise ValueError('Invalid Studio job ID')
    return Path(base) / identifier


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def glb_metadata(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic, version, length = struct.unpack('<4sII', stream.read(12))
        if magic != b'glTF' or version != 2 or length != path.stat().st_size:
            raise ValueError('Incomplete GLB 2.0 output')
        size, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or size > 16 * 1024**2:
            raise ValueError('Invalid GLB metadata')
        model = json.loads(stream.read(size))
    vertices = triangles = 0
    for mesh in model.get('meshes', []):
        for primitive in mesh['primitives']:
            vertices += model['accessors'][primitive['attributes']['POSITION']]['count']
            if 'indices' in primitive:
                triangles += model['accessors'][primitive['indices']]['count'] // 3
    if not vertices or not triangles:
        raise ValueError('The GLB contains no mesh')
    return dict(bytes=length, sha256=sha256_file(path), vertices=vertices,
                triangles=triangles, texture_size=2048)
