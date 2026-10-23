#!/bin/sh
set -eu
TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_REPO"
echo 'UniForge RTX 3090 GPUQ Studio: http://100.85.200.56:7864'
echo 'Generation uses only the existing .51:12434 proxy and its shared GPUQ.'
exec studio/.venv/bin/python -m uvicorn studio.server.cuda3090:app --host 0.0.0.0 --port 7864 --no-access-log
