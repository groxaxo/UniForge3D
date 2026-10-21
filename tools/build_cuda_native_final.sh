#!/bin/bash
set -euxo pipefail
cd /home/op/uniforge3d-work/UniForge3D
export CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/op/uniforge3d-work/env/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 PIP_NO_CACHE_DIR=1
export TMPDIR=/home/op/uniforge3d-work/tmp
python -m pip install plyfile zstandard
python -m pip install ./.deps/TRELLIS.2/o-voxel --no-build-isolation --no-deps
python -c 'import torch, flash_attn, cumesh, flex_gemm, nvdiffrast, o_voxel; print(torch.__version__, torch.cuda.get_device_name(0))'
echo NATIVE_ENV_READY
