"""Package the tracked CUDA sources without copying weights or changing a host."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import tarfile


def package(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files',
        'uniforge3d', 'tools/run_cuda_staged.py'], text=True).splitlines()
    manifest = {'source_git_head': subprocess.check_output(['git', '-C', str(root),
        'rev-parse', 'HEAD'], text=True).strip(), 'files': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    raw = json.dumps(manifest, indent=2).encode()
    identifier = hashlib.sha256(raw).hexdigest()[:16]
    (output / 'runtime-id').write_text(identifier)
    with tarfile.open(output / 'runtime.tar.gz', 'w:gz') as archive:
        for name in files:
            archive.add(root / name, arcname=name)
        entry = tarfile.TarInfo('manifest.json')
        entry.size, entry.mode = len(raw), 0o644
        archive.addfile(entry, io.BytesIO(raw))
    with tarfile.open(output / 'bootstrap.tar.gz', 'w:gz') as archive:
        for name in ['studio/server/gpuq_common.py', 'studio/server/gpuq_gateway.py',
                     'studio/server/gpuq_worker.py', 'tools/configure_uniforge_proxy.py']:
            archive.add(root / name, arcname=name)
    return {'runtime_id': identifier, 'source_files': len(files), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/deployment/3090'))
    print(json.dumps(package(parser.parse_args().output)))
