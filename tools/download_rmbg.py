"""Read the existing HF token from stdin; download only the pinned RMBG files."""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
secret=sys.stdin.readline().strip()
assert secret, 'No existing HF token received'
pipe=Path('/home/op/uniforge3d-work/rmbg-token.pipe')
pipe.unlink(missing_ok=True)
snapshot_download('briaai/RMBG-2.0',revision='5df4c9c76d8170882c34f6986e848ee07fd0ba43',
    local_dir='/home/op/uniforge3d-work/weights/rmbg',token=secret,
    allow_patterns=['BiRefNet_config.py','birefnet.py','config.json','model.safetensors','preprocessor_config.json','README.md'])
print('RMBG_READY',flush=True)
