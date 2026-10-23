"""Verified checkpoint inventory; optional SAM choices are reference metadata only."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import threading

SOURCE_REPO = 'Comfy-Org/sam-3d-body'
SOURCE_REVISION = '7c974aea58e062252a6cb5b6f9b7f639d545b729'
# Published Hugging Face LFS SHA256/size at SOURCE_REVISION; no weights are bundled.
OPTIONAL_MODELS = (
    {'id': 'sam-3d-body-bf16', 'label': 'SAM 3D Body · BF16',
     'file': 'detection/sam_3d_body_dinov3_bf16.safetensors', 'bytes': 2830737652,
     'sha256': '59fa45200c504c5b56625004d7d3385daf48c616613e88099e43bf83b3e249cf'},
    {'id': 'sam-3d-body-int8', 'label': 'SAM 3D Body · INT8 ConvRot',
     'file': 'detection/sam_3d_body_dinov3_int8_convrot.safetensors', 'bytes': 1994073596,
     'sha256': 'ba8d4c34a81bc072b3f53db5241074d370b26e3b72ae793630be5344efd28934'},
)
_verification_lock = threading.Lock()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@lru_cache(maxsize=16)
def _checkpoint_hash(filename, identity):
    """Hash each stable file once; reject partial/preallocated headers cheaply."""
    path = Path(filename)
    with path.open('rb') as stream:
        if _identity(os.fstat(stream.fileno())) != identity:
            return None
        prefix = stream.read(8)
        if len(prefix) != 8:
            return None
        header_size = struct.unpack('<Q', prefix)[0]
        if not 2 <= header_size <= min(4 * 1024**2, identity[2] - 8):
            return None
        raw_header = stream.read(header_size)
        try:
            header = json.loads(raw_header)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(header, dict) or not any(name != '__metadata__' for name in header):
            return None
        digest = hashlib.sha256(prefix + raw_header)
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
        if (_identity(os.fstat(stream.fileno())) != identity
                or _identity(path.stat()) != identity):
            return None
    return digest.hexdigest()


def checkpoint_status(path, spec):
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            return 'invalid'
        if info.st_size != spec['bytes']:
            return 'incomplete'
        with _verification_lock:
            digest = _checkpoint_hash(str(path.resolve()), _identity(info))
        return 'verified' if digest == spec['sha256'] else 'invalid'
    except FileNotFoundError:
        return 'missing'
    except OSError:
        return 'invalid'


def optional_model_record(identifier):
    spec = next(item for item in OPTIONAL_MODELS if item['id'] == identifier)
    return {'id': identifier, 'label': spec['label'], 'expected_sha256': spec['sha256'],
            'repository': SOURCE_REPO, 'revision': SOURCE_REVISION,
            'usage': 'reference-only', 'executed': False}


def model_catalog(models_dir):
    """A selectable SAM file must match its complete published SHA256."""
    models = [{'id': 'uniforge3d', 'label': 'UniForge3D', 'optional': False, 'installed': True}]
    for spec in OPTIONAL_MODELS:
        status = checkpoint_status(Path(models_dir) / spec['file'], spec)
        models.append({'id': spec['id'], 'label': spec['label'], 'optional': True,
                       'installed': status == 'verified', 'verification': status,
                       'bytes': spec['bytes'], 'sha256': spec['sha256'],
                       'usage': 'reference-only', 'executed': False})
    return models
