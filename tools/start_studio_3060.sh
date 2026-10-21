#!/bin/sh
set -eu
TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_REPO"
echo 'UniForge RTX 3060 Studio: http://127.0.0.1:7863 (network: http://100.85.200.56:7863; bind 0.0.0.0:7863)'
echo 'CUDA compute: op@100.85.200.54. Source environment/weights: read-only SSHFS from .51.'
exec studio/.venv/bin/python -m uvicorn studio.server.cuda3060:app --host 0.0.0.0 --port 7863 --no-access-log
