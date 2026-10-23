"""Install the narrowly scoped UniForge integration into the existing .51 proxy.

Run on .51 after deploying studio/server/gpuq_*.py and its pinned CUDA runtime.
This preserves every other family, makes backups, and never starts GPU work.
Restart the existing proxy only after checking its live requests and owners.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import time


def configure(root, repo):
    root, repo = Path(root), Path(repo)
    proxy = root / 'qwen36_variant_proxy.py'
    registry = root / 'qwen36_model_families.json'
    if proxy.is_symlink() or registry.is_symlink():
        raise RuntimeError('Review symlinked proxy configuration before updating it')
    source = proxy.read_text()
    families = json.loads(registry.read_text())
    family = dict(enabled=True, name='UniForge3D · RTX 3090 · .51 · GPUQ',
        backend_model='uniforge3d', public_model_id='uniforge3d', task='image-to-3d',
        backend='gpuq', queue_mode='per-job', backend_managed=False,
        backend_base_url='http://127.0.0.1:12434', backend_ready_path='/v1/3d/health',
        model_path='/home/op/uniforge3d-work/weights/dino-hf', variants=['general'],
        backend_required_files=[str(repo / 'studio/server' / name)
            for name in ('gpuq_gateway.py', 'gpuq_worker.py', 'gpuq_common.py')]
            + ['/home/op/uniforge3d-work/studio-gpuq/runtime/manifest.json'],
        description='Asynchronous image-to-GLB generation; one flexible GPUQ GPU per job. Source CUDA precision and original DINOv3.')
    if 'uniforge3d' in families and families['uniforge3d'] != family:
        raise RuntimeError('An existing UniForge family differs; review it before replacing it')
    install = (f'\n# UniForge owns durable per-job GPUQ admission; these routes are CPU-only.\n'
               f'sys.path.insert(0, {str(repo)!r})\n'
               'from studio.server.gpuq_gateway import register as register_uniforge\n'
               'register_uniforge(app)\n')
    anchor = 'app = FastAPI(title="qwen36-multi-variant-proxy")\n'
    if 'register_uniforge(app)' not in source:
        if source.count(anchor) != 1:
            raise RuntimeError('Expected proxy application anchor was not found exactly once')
        source = source.replace(anchor, anchor + install, 1)
    catalog = '''        if config.get("task") == "image-to-3d":
            data.append({"id": config.get("public_model_id", prefix), "object": "model",
                "owned_by": "local", "family_id": prefix, "name": config["name"],
                "description": config.get("description", "GPUQ image-to-GLB generation"),
                "task": "image-to-3d", "backend": "gpuq", "queue_mode": "per-job",
                "reasoning": False, "temperature": False, "tool_call": False,
                "attachment": True, "open_weights": True,
                "modalities": {"input": ["image"], "output": ["mesh"]},
                "route": "POST /v1/3d/jobs/{id}"})
            continue
'''
    anchor = ('def list_models() -> dict[str, Any]:\n    data = []\n'
              '    for prefix, config in _available_families().items():\n')
    if catalog not in source:
        if source.count(anchor) != 1:
            raise RuntimeError('Expected model catalog anchor was not found exactly once')
        source = source.replace(anchor, anchor + catalog, 1)
    compile(source, str(proxy), 'exec')
    families['uniforge3d'] = family
    backup = root / 'backups' / f'uniforge-gpuq-{time.time_ns()}'
    changes = {proxy: source, registry: json.dumps(families, indent=2) + '\n'}
    changed = {path: text for path, text in changes.items() if path.read_text() != text}
    if changed:
        backup.mkdir(parents=True, mode=0o700)
        for path, text in changed.items():
            mode = stat.S_IMODE(path.stat().st_mode)
            shutil.copy2(path, backup / path.name)
            temporary = path.with_suffix(path.suffix + '.uniforge.tmp')
            temporary.write_text(text)
            os.chmod(temporary, mode)
            temporary.replace(path)
    return {'changed': [path.name for path in changed], 'backup': str(backup) if changed else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy-root', type=Path, required=True)
    parser.add_argument('--runtime-repo', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(configure(args.proxy_root, args.runtime_repo)))
