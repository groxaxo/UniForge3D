#!/bin/bash
set -euxo pipefail
cd /home/op/uniforge3d-work/UniForge3D
export CUDA_VISIBLE_DEVICES=GPU-78996a05-18c5-e153-b621-096273299d41
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export PIP_NO_CACHE_DIR=1
export TMPDIR=/home/op/uniforge3d-work/tmp
mkdir -p "$TMPDIR"
/home/op/miniforge3/bin/conda create -p /home/op/uniforge3d-work/env python=3.10 pip -y
export PATH=/home/op/uniforge3d-work/env/bin:$PATH
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install setuptools==80.9.0 wheel ninja cmake==3.31.6 packaging transformers==4.57.3 peft==0.17.1 'huggingface-hub>=0.36,<1' 'timm>=1.0,<2' 'kornia>=0.8,<0.9' 'numpy>=2.1,<2.3' 'Pillow>=10.4,<12' 'safetensors>=0.7,<1' trimesh==4.6.3 'opencv-python-headless>=4.10,<5' 'easydict>=1.13,<2' 'tqdm>=4.66,<5' 'scipy>=1.13,<2'
echo BASE_ENV_READY
