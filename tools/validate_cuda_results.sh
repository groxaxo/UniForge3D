#!/bin/bash
set -euo pipefail
cd /home/op/uniforge3d-work/UniForge3D
export CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/op/uniforge3d-work/env/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4
OUT=/home/op/quant-jobs/holo4-exl3-6bpw-8bpw-20261009/storage-wsl/uniforge3d-20261020/outputs
while ! grep -q BOTH_RUNS_COMPLETE /home/op/uniforge3d-work/logs/mission.log; do sleep 2; done
python -u tools/inspect_glb.py "$OUT/case01/1a5ab5b40117/prediction.glb"
python -u tools/inspect_glb.py "$OUT/facu-20261020/f3f62fc827c9/prediction.glb"
python tools/collect_run_results.py "$OUT" --gpu-log /home/op/uniforge3d-work/logs/inference-gpu.csv > /home/op/uniforge3d-work/logs/run_results.json
echo VALIDATION_COMPLETE
