#!/bin/sh
set -eu
TASK_REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_REPO"
if [ ! -x studio/.venv/bin/python ] || [ ! -f studio/web/dist/index.html ]; then
  echo 'Studio needs its local environment and frontend build. See studio/README.md.' >&2
  exit 1
fi
echo 'UniForge Studio: http://127.0.0.1:7861 (network: http://100.85.200.56:7861; bind 0.0.0.0:7861)'
echo 'This local UI runs generation on this Mac with the verified MLX bundle.'
exec studio/.venv/bin/python -m uvicorn studio.server.app:app --host 0.0.0.0 --port 7861 --no-access-log
