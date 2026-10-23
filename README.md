# UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space

Wenjue Chen<sup>1,&#42;</sup>, Wenyue Chen<sup>1,2,&#42;</sup>, Heliang Zheng<sup>2</sup>, Rongfei Jia<sup>2</sup>, Yuan Liu<sup>3,&dagger;</sup>, Meng Li<sup>1,&dagger;</sup>

<sup>1</sup> Peking University &nbsp; <sup>2</sup> Math Magic &nbsp; <sup>3</sup> The Hong Kong University of Science and Technology

<sup>&#42;</sup> Equal contribution. <sup>&dagger;</sup> Corresponding authors.

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=flat&logo=googlechrome&logoColor=white)](https://xishuxishu.github.io/uniforge3d-project/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?style=flat&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/xibushu/UniForge3D)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.15357-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.15357)

![UniForge3D teaser](assets/teaser.png)

UniForge3D generates complete 3D assets from sparse, unposed multi-view images. It jointly models reconstruction and generation in a shared anchor-guided space, then uses structure-guided view selection to preserve observed details. 

## Installation

```bash
git clone https://github.com/xibushu/UniForge3D.git
cd UniForge3D
bash setup.sh
conda activate uniforge3d
```

## Inference

The [Android Studio client](android/README.md) opens the same interface on the
Moto G54 over Tailscale, with native image picking and downloads. Build its signed
APK with `./tools/build_android.sh`, then install using
`./tools/install_android_g54.sh`. Generation continues on the selected Mac or CUDA
backend.

For the local browser interface on this Mac, double-click
`Open UniForge Studio.command` or run `./tools/studio_service.sh install` and open
**http://127.0.0.1:7861** on the Mac or **http://100.85.200.56:7861** from another Tailscale device. The frontend and API listen on **0.0.0.0:7861**. [UniForge Studio](studio/README.md) provides one-to-four-view
uploads, a job queue, an interactive 3D preview, and GLB downloads through the
verified local MLX runtime. New jobs use the selective oQ5 bundle and original
DINOv3 encoder on this Mac.

Selective MLX/oMLX oQ5 checkpoint conversion tools and their current validation
status are documented in [MLX_QUANTIZATION.md](MLX_QUANTIZATION.md). The inference
command below is the original CUDA pipeline.

The recommended [complete MLX model bundle](ONNX_DINOV3.md) now includes all
15 components, using the public original-model DINOv3 ONNX export with tested
FP16 vision projections. The public source downloads without gated access.
The [local MLX runtime](MLX_RUNTIME.md) now runs the full image-to-textured-GLB
pipeline on Apple Silicon:

```sh
.venv/bin/python tools/run_mlx.py assets/case02/001.png \
  --output run_logs/my-mlx-run
```

The validated single-view example produced a 995,066-triangle GLB with 2048px
PBR textures. The runtime documentation includes numerical checks, the
rendered result, timings, multiview validation scope and remaining CUDA-parity
limits. Studio now uses this runtime by default; its older private CUDA engine
is retained as an optional backend.

The user-selected public SoftEq DINOv3 replacement, its encoder validation, and
the choice to protect vision weights at FP16 are documented in
[SOFTEQ_DINOV3.md](SOFTEQ_DINOV3.md).

For face-photo texture correction after MLX export, the tested
[GLB compatibility step](FACE_COMPATIBILITY.md) consolidates same-material
primitive chunks without changing geometry, UVs or textures, then feeds the
pinned face-retexture tool. It includes regression and complete CPU vision checks.

Place the views of each object directly in a separate folder under `assets/`:

```text
assets/
├── case01/
│   ├── 001.png
│   ├── 002.png
│   ├── 003.png
│   └── 004.png
└── case02/
    └── ...
```

Run all cases using the pretrained model:

```bash
python infer.py
```



## Acknowledgements

Our code is based on these wonderful works:

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [VGGT](https://github.com/facebookresearch/vggt)

## License

Original UniForge3D source code is released under the
[MIT License](LICENSE). Third-party code, dependencies, and model
weights retain their respective licenses and are not relicensed
under this project's MIT License. Please refer to their original
license files and notices for applicable terms.

The MIT License for our source code does not apply to the released
model weights. In particular, our VGGT-derived checkpoints were
initialized from
[facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
and remain subject to
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The default background remover,
[BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0),
requires separate authorization for commercial use.

## Citation

If you find UniForge3D useful in your research, please cite our [paper](https://arxiv.org/abs/2610.15357):

```bibtex
@article{chen2026uniforge3d,
  title   = {UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space},
  author  = {Chen, Wenjue and Chen, Wenyue and Zheng, Heliang and Jia, Rongfei and Liu, Yuan and Li, Meng},
  journal = {arXiv preprint arXiv:2610.15357},
  year    = {2026}
}
```

