#!/bin/bash
set -euo pipefail
cd /home/op/uniforge3d-work/UniForge3D
export CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/op/uniforge3d-work/env/bin:$CUDA_HOME/bin:$PATH
export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn SPARSE_CONV_BACKEND=flex_gemm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6
OUT=/home/op/quant-jobs/holo4-exl3-6bpw-8bpw-20261009/storage-wsl/uniforge3d-20261020/outputs
mkdir -p "$OUT"
while ! grep -q NATIVE_ENV_READY /home/op/uniforge3d-work/logs/native.log || ! grep -q PUBLIC_WEIGHTS_READY /home/op/uniforge3d-work/logs/weights.log || ! grep -q RMBG_READY /home/op/uniforge3d-work/logs/rmbg-download.log; do
 if [ -f /home/op/uniforge3d-work/logs/native.exit ] && ! grep -qx 0 /home/op/uniforge3d-work/logs/native.exit; then echo NATIVE_BUILD_FAILED; exit 1; fi
 sleep 5
done
nvidia-smi --id=GPU-78996a05-18c5-e153-b621-096273299d41 --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > /home/op/uniforge3d-work/logs/inference-gpu.csv &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python -u tools/run_cuda_staged.py --input assets/case01 --output "$OUT"
python -u tools/run_cuda_staged.py --input assets/facu-20261020 --output "$OUT"
echo BOTH_RUNS_COMPLETE
