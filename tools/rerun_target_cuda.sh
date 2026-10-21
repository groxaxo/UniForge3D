#!/usr/bin/env bash
# Run from the Mac after replacing assets/facu-20261020/{001,002,003,004}.png.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
REMOTE=op@100.85.200.51
REMOTE_REPO=/home/op/uniforge3d-work/UniForge3D
REMOTE_OUT=/home/op/quant-jobs/holo4-exl3-6bpw-8bpw-20261009/storage-wsl/uniforge3d-20261020/outputs
RUN_ID="uniforge3d-target-$(date -u +%Y%m%dT%H%M%SZ)"
for VIEW in 001 002 003 004; do test -s "assets/facu-20261020/$VIEW.png"; done
# Do not start a competing GPU workload. Existing services are never changed.
GPU_USED=$(ssh "$REMOTE" 'nvidia-smi --id=GPU-78996a05-18c5-e153-b621-096273299d41 --query-gpu=memory.used --format=csv,noheader,nounits')
if [ "$GPU_USED" -gt 100 ]; then echo "GPU 2 is occupied ($GPU_USED MiB); rerun when it is free." >&2; exit 1; fi
COPYFILE_DISABLE=1 tar -cf - assets/facu-20261020/001.png assets/facu-20261020/002.png assets/facu-20261020/003.png assets/facu-20261020/004.png |
    ssh "$REMOTE" "tar -xf - -C $REMOTE_REPO"
ssh "$REMOTE" "tmux new-session -d -s $RUN_ID 'cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41 CUDA_HOME=/usr/local/cuda-12.8 /home/op/uniforge3d-work/env/bin/python -u tools/run_cuda_staged.py --input assets/facu-20261020 --output $REMOTE_OUT > /home/op/uniforge3d-work/logs/$RUN_ID.log 2>&1'"
echo "Started $REMOTE tmux session $RUN_ID"
echo "Log: /home/op/uniforge3d-work/logs/$RUN_ID.log"
echo "The log's GLB_COMPLETE line gives the input-hash-specific GLB path."
echo "Resume with the same Python command if a stage is interrupted."
