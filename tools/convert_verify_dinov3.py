"""Convert pinned timm DINOv3 ViT-L/16 to HF and gate use on numerical parity.

Example:
CUDA_VISIBLE_DEVICES=<GPU-UUID> python tools/convert_verify_dinov3.py \
 --source /home/op/uniforge3d-work/weights/dino-timm/model.safetensors \
 --output /home/op/uniforge3d-work/weights/dino-hf --image assets/case01/001.png

No downloads, no source mutations. All CLS, 4 register, and patch tokens are tested.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


def convert(source):
    converted = {}
    for key, value in source.items():
        direct = {'cls_token': 'embeddings.cls_token', 'reg_token': 'embeddings.register_tokens',
                  'patch_embed.proj.weight': 'embeddings.patch_embeddings.weight',
                  'patch_embed.proj.bias': 'embeddings.patch_embeddings.bias',
                  'norm.weight': 'norm.weight', 'norm.bias': 'norm.bias'}
        if key in direct:
            converted[direct[key]] = value
            continue
        match = re.fullmatch(r'blocks\.(\d+)\.(.*)', key)
        if not match:
            raise ValueError(f'Unmapped timm parameter: {key}')
        index, suffix = match.groups()
        prefix = f'layer.{index}.'
        if suffix == 'attn.qkv.weight':
            for name, tensor in zip(('q', 'k', 'v'), value.chunk(3, dim=0)):
                converted[prefix + f'attention.{name}_proj.weight'] = tensor.contiguous()
        elif suffix in ('attn.q_bias', 'attn.v_bias'):
            converted[prefix + f'attention.{suffix[5]}_proj.bias'] = value
        else:
            suffix = suffix.replace('attn.proj.', 'attention.o_proj.')
            suffix = suffix.replace('mlp.fc1.', 'mlp.up_proj.').replace('mlp.fc2.', 'mlp.down_proj.')
            suffix = {'gamma_1': 'layer_scale1.lambda1', 'gamma_2': 'layer_scale2.lambda1'}.get(suffix, suffix)
            if not re.fullmatch(r'(norm[12]|attention.o_proj|mlp.(up_proj|down_proj))\.(weight|bias)|layer_scale[12].lambda1', suffix):
                raise ValueError(f'Unmapped block parameter: {key}')
            converted[prefix + suffix] = value
    # The distilled L/16 timm checkpoint omits exactly-zero q/v biases and the
    # training-only mask token. HF represents these explicitly.
    converted['embeddings.mask_token'] = torch.zeros(1, 1, 1024)
    for i in range(24):
        for name in ('q', 'v'):
            converted.setdefault(f'layer.{i}.attention.{name}_proj.bias', torch.zeros(1024))
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = load_file(str(args.source))
    reference = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=False)
    reference.load_state_dict(source, strict=True)
    reference.eval().requires_grad_(False).cuda()
    config = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16, image_size=256,
        num_register_tokens=4, query_bias=True, key_bias=False, value_bias=True,
        layerscale_value=1e-5, layer_norm_eps=1e-5, rope_theta=100.0)
    config._attn_implementation = 'sdpa'
    model = DINOv3ViTModel(config)
    model.load_state_dict(convert(source), strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    del model, source
    model = DINOv3ViTModel.from_pretrained(args.output, local_files_only=True,
                                          attn_implementation='sdpa').eval().requires_grad_(False).cuda()
    image = Image.open(args.image).convert('RGB')
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    result = {'source_repo':'timm/vit_large_patch16_dinov3.lvd1689m',
        'source_revision':'30c1109559f65dea34316b0d4842d35c5771fe11',
        'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'image':str(args.image), 'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'timm':timm.__version__, 'torch':torch.__version__, 'tests':[]}
    with torch.inference_mode():
        for size in args.sizes:
            start = time.perf_counter()
            arr=np.array(image.resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/255.
            pixel=(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).cuda()-mean)/std
            expected=reference.forward_features(pixel).float()
            actual=model(pixel_values=pixel).last_hidden_state.float()
            wanted=(1, 1+4+(size//16)**2, 1024)
            assert tuple(actual.shape)==tuple(expected.shape)==wanted
            cosine=F.cosine_similarity(expected.flatten(),actual.flatten(),dim=0).item()
            min_token_cosine=F.cosine_similarity(expected,actual,dim=-1).min().item()
            test={'size':size,'timm_shape':list(expected.shape),'hf_shape':list(actual.shape),
                'cosine':cosine,'minimum_token_cosine':min_token_cosine,
                'max_abs_error':(expected-actual).abs().max().item(),
                'mean_abs_error':(expected-actual).abs().mean().item(),
                'seconds':time.perf_counter()-start}
            result['tests'].append(test)
            print(json.dumps(test),flush=True)
            assert cosine>=0.999 and min_token_cosine>=0.999, 'DINOv3 parity gate failed'
    result['verified']=True
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DINO_PARITY_VERIFIED',flush=True)

if __name__ == '__main__':
    main()
