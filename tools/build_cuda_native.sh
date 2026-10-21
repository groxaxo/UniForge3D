#!/bin/bash
set -euxo pipefail
cd /home/op/uniforge3d-work/UniForge3D
export CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/op/uniforge3d-work/env/bin:$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 PIP_NO_CACHE_DIR=1
export TMPDIR=/home/op/uniforge3d-work/tmp
mkdir -p .deps
clone_pin() {
 name="$1"; url="$2"; rev="$3"
 if [ ! -d ".deps/$name/.git" ]; then
  git init ".deps/$name"
  git -C ".deps/$name" remote add origin "$url"
  git -C ".deps/$name" fetch --depth 1 origin "$rev"
  git -C ".deps/$name" checkout FETCH_HEAD
  git -C ".deps/$name" submodule update --init --recursive --depth 1
 fi
}
clone_pin TRELLIS.2 https://github.com/microsoft/TRELLIS.2.git 75fbf0183001ed9876c8dbb35de6b68552ee08bd
clone_pin CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f
clone_pin FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e
clone_pin nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0
until grep -q BASE_ENV_READY /home/op/uniforge3d-work/logs/build.log; do sleep 5; done
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install ./.deps/nvdiffrast --no-build-isolation
python -m pip install ./.deps/CuMesh --no-build-isolation
python -m pip install ./.deps/FlexGEMM --no-build-isolation
python -m pip install plyfile zstandard
python -m pip install ./.deps/TRELLIS.2/o-voxel --no-build-isolation --no-deps
python -c 'import torch, flash_attn, cumesh, flex_gemm, nvdiffrast, o_voxel; print(torch.__version__, torch.cuda.get_device_name(0))'
echo NATIVE_ENV_READY
