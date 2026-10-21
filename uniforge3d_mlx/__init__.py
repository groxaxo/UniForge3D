"""MLX checkpoint conversion and inference for UniForge3D.

This package deliberately does not import the CUDA-only inference package.
"""
import os

# Apple M5's optional TF32 products lose more precision than the checkpoints'
# FP32 geometry/conditioning boundaries. Set before the first Metal matmul.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
