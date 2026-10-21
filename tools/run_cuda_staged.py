"""Run the original UniForge3D stages in isolated processes to bound host RAM.

Replacement images are detected by SHA256; a fresh run folder is chosen per
input/config fingerprint. Run under tmux. No remote Hub access at inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
from datetime import datetime, timezone
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASES = ['preprocess', 'dino', 'structure', 'shape512', 'shape1024', 'texture', 'decode']


def input_images(directory):
    images = sorted(directory.glob('*.png'))
    expected = [directory / f'{i:03d}.png' for i in range(1, len(images) + 1)]
    if not 1 <= len(images) <= 4 or images != expected:
        raise ValueError('Expected one to four ordered input files named 001.png through 004.png')
    return images


def make_config(weights):
    config = json.loads((ROOT / 'uniforge3d/configs/pipeline.json').read_text())
    def resolve(value):
        if isinstance(value, dict) and 'repo_id' in value and 'filename' in value:
            group = 'trellis1' if value['repo_id'] == 'microsoft/TRELLIS-image-large' else 'trellis2'
            return str(weights / group / value['filename'])
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str) and value.startswith('uniforge3d/'):
            return str(weights / 'uniforge3d' / value)
        return value
    config = resolve(config)
    config['external_models']['dino'] = str(weights / 'dino-hf')
    config['external_models']['rembg'] = str(weights / 'rmbg')
    config['image_preprocess']['rembg_model'] = str(weights / 'rmbg')
    config['conditioner']['vggt_feature_extractor']['checkpoint_path'] = str(
        weights / 'uniforge3d/uniforge3d/vggt/uniforge3d_shape_texture_vggt_conditioner.pt')
    return config


def worker(args):
    import torch
    import trimesh
    from easydict import EasyDict as edict
    from uniforge3d import UniForge3DImageTo3DPipeline as Pipeline
    from uniforge3d.models import loaders
    from uniforge3d.models.conditioning import StructuredLatentConditioner
    from uniforge3d.models.dino import DinoV3Encoder
    from uniforge3d.pipelines.types import SparseStructure, NormalizedShapeLatent
    from uniforge3d.pipelines.texture import (denormalize_shape_latent, sample_texture,
        decode_shape_and_texture, remesh_and_bake_texture)
    from uniforge3d.modules.sparse import SparseTensor
    from uniforge3d.representations.mesh import Mesh
    from uniforge3d.utils.preprocessing import BackgroundRemover, InputBatch
    from uniforge3d.utils.mesh import export_glb
    from uniforge3d.utils.device import release_cuda
    from uniforge3d.utils.mesh_budget import cpu_reduce_if_needed

    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    config = json.loads((args.work / 'config.json').read_text())
    p = Pipeline.__new__(Pipeline)
    p.config, p.device, p.low_vram = config, torch.device('cuda', 0), True
    load = lambda name: torch.load(args.work / name, map_location='cpu', weights_only=True)
    save = lambda obj, name: torch.save(obj, args.work / name)
    def restore_rng():
        previous = PHASES.index(args.phase) - 1
        if previous >= 0 and (args.work / f'{PHASES[previous]}.rng.pt').exists():
            state = load(f'{PHASES[previous]}.rng.pt')
            torch.set_rng_state(state['cpu'])
            torch.cuda.set_rng_state(state['cuda'])
        else:
            torch.manual_seed(config['default_seed'])
    def drop(*names):
        for name in names:
            if hasattr(p, name): delattr(p, name)
        release_cuda()
    def flow(stage):
        spec = edict(config[stage])
        return loaders._load_flow_model(spec.model, spec.weights)

    with torch.no_grad():
        if args.phase == 'preprocess':
            p.remove_background = BackgroundRemover(config['image_preprocess']['rembg_model'], p.device,
                                                    local_files_only=True)
            restore_rng()
            images = input_images(args.input)
            batch = p._preprocess_images(images)
            save(vars(batch), 'batch.pt')
        else:
            batch = InputBatch(**load('batch.pt'))
            if args.phase == 'dino':
                p.dino = DinoV3Encoder(config['external_models']['dino'], local_files_only=True)
                restore_rng()
                save(p._encode_dino(batch), 'dino.pt')
            elif args.phase == 'structure':
                spec = edict(config['sparse_structure'])
                p.structure_model = loaders.get_model_class(spec.model.name)(**spec.model.args)
                loaders._load_structure_weights(p.structure_model, spec.weights)
                p.structure_decoder = loaders.load_sparse_structure_decoder(spec.decoder.checkpoint)
                restore_rng()
                output = p._generate_structure(batch, load('dino.pt')['sparse_structure'])
                save(vars(output), 'structure.pt')
            elif args.phase == 'shape512':
                spec = edict(config['structured_latent_512'])
                p.shape512_model = flow('structured_latent_512')
                p.shape512_decoder = loaders.load_shape_decoder(spec.decoder.checkpoint, 512)
                p.conditioner = StructuredLatentConditioner(**config['conditioner'])
                restore_rng()
                mesh = p._generate_shape512(batch, SparseStructure(**load('structure.pt')), load('dino.pt')['shape512'])
                mesh.export(args.work / 'shape512.ply')
                save(p._shape512_to_coords64(mesh), 'coords64.pt')
            elif args.phase == 'shape1024':
                p.shape1024_model = flow('structured_latent_1024')
                p.conditioner = StructuredLatentConditioner(**config['conditioner'])
                restore_rng()
                latent, view_assignment = p._generate_shape1024(batch, load('coords64.pt'), load('dino.pt')['shape1024'])
                save({'coords':latent.coords, 'feats':latent.feats, 'view_assignment':view_assignment}, 'shape1024.pt')
            elif args.phase == 'texture':
                p.texture_model = flow('texture')
                p.conditioner = StructuredLatentConditioner(**config['conditioner'])
                shape = load('shape1024.pt')
                shape_latent = denormalize_shape_latent(config['texture'],
                    NormalizedShapeLatent(shape['coords'], shape['feats']), p.device)
                restore_rng()
                with p._use_models(p.texture_model, p.conditioner):
                    texture = sample_texture(p.texture_model, config['texture'], p.conditioner,
                        shape_latent, batch, load('dino.pt')['shape1024'], shape['view_assignment'], p.device)
                save({'coords':texture.coords.cpu(), 'feats':texture.feats.cpu()}, 'texture.pt')
            elif args.phase == 'decode':
                small_gpu = torch.cuda.get_device_properties(0).total_memory <= 13 * 1024**3
                cpu_deps = ROOT.parent / 'cpu-mesh-deps'
                if cpu_deps.exists():
                    sys.path.insert(0, str(cpu_deps))
                face_budget = 4_000_000 if small_gpu else None
                signature = {name: hashlib.sha256((args.work / name).read_bytes()).hexdigest()
                             for name in ['config.json', 'inputs.json', 'shape1024.pt', 'texture.pt']}
                decoded_cache = args.work / 'decoded.pt'
                if decoded_cache.exists():
                    cached = load('decoded.pt')
                    if cached['signature'] != signature:
                        raise RuntimeError('Decoded checkpoint no longer matches its inputs and latent checkpoints')
                    mesh = Mesh(cached['vertices'].to(p.device), cached['faces'].to(p.device))
                    voxel = SparseTensor(feats=cached['feats'], coords=cached['coords']).to(p.device)
                    torch.set_rng_state(cached['cpu_rng'])
                    torch.cuda.set_rng_state(cached['cuda_rng'])
                    del cached
                    release_cuda()
                    print('Reusing verified raw decoder checkpoint', flush=True)
                else:
                    p.shape1024_decoder = loaders.load_shape_decoder(config['texture']['shape_decoder_checkpoint'],1024)
                    p.texture_decoder = loaders.load_texture_decoder(config['texture']['texture_decoder_checkpoint'])
                    shape = load('shape1024.pt')
                    shape_latent = denormalize_shape_latent(config['texture'],
                        NormalizedShapeLatent(shape['coords'], shape['feats']), p.device)
                    tex = load('texture.pt')
                    texture = SparseTensor(feats=tex['feats'], coords=tex['coords']).to(p.device)
                    restore_rng()
                    with p._use_models(p.shape1024_decoder, p.texture_decoder):
                        mesh, voxel = decode_shape_and_texture(p.shape1024_decoder, p.texture_decoder, shape_latent, texture,
                                                              fill_holes=False)
                    voxel.clear_spatial_cache()
                    del shape_latent, texture, tex, shape
                    drop('shape1024_decoder','texture_decoder')
                    temporary = args.work / 'decoded.pt.partial'
                    torch.save(dict(signature=signature, vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(),
                                    feats=voxel.feats.cpu(), coords=voxel.coords.cpu(),
                                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temporary)
                    temporary.replace(decoded_cache)
                print(json.dumps({'event':'decoders_released_before_mesh',
                                  'allocated_bytes':torch.cuda.memory_allocated(),
                                  'raw_vertices':len(mesh.vertices), 'raw_faces':len(mesh.faces),
                                  'cuda_face_budget':face_budget}), flush=True)
                mesh.vertices, mesh.faces = cpu_reduce_if_needed(mesh.vertices, mesh.faces, face_budget)
                release_cuda()
                mesh.fill_holes()
                release_cuda()
                mesh = remesh_and_bake_texture(config['texture'], config['mesh_postprocess'], mesh, voxel,
                                              cuda_face_budget=face_budget)
                export_glb(mesh, args.work / 'prediction.glb', overwrite=True)
        save({'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state()}, f'{args.phase}.rng.pt')
        torch.cuda.synchronize()
        result = {'phase':args.phase, 'started_at':started_at, 'finished_at':datetime.now(timezone.utc).isoformat(), 'seconds':time.perf_counter()-start,
            'gpu_name':torch.cuda.get_device_name(0),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if args.phase == 'decode':
            result['cuda_face_budget'] = face_budget
            result['cpu_mesh_dependency'] = 'fast-simplification==0.1.13' if face_budget else None
        (args.work / f'{args.phase}.done.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('/home/op/uniforge3d-work/weights'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if args.phase:
        worker(args)
        return
    verification = json.loads((args.weights / 'dino-hf/verification.json').read_text())
    assert verification['verified'] and all(t['cosine'] >= 0.999 for t in verification['tests'])
    config = make_config(args.weights)
    images = input_images(args.input)
    provenance = {'config':config, 'images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images}}
    fingerprint = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:12]
    work = args.output / args.input.name / fingerprint
    work.mkdir(parents=True,exist_ok=True)
    (work/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (work/'inputs.json').write_text(json.dumps(provenance['images'],indent=2)+'\n')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ATTN_BACKEND='flash_attn',
        SPARSE_ATTN_BACKEND='flash_attn', SPARSE_CONV_BACKEND='flex_gemm')
    for phase in PHASES:
        if (work/f'{phase}.done.json').exists():
            print(f'Resume: {phase} already complete',flush=True)
            continue
        print(f'Starting {phase}: {work}',flush=True)
        with (work/f'{phase}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--input',str(args.input),
                '--output',str(args.output),'--work',str(work),'--phase',phase],
                stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        print((work/f'{phase}.done.json').read_text(),flush=True)
    print('GLB_COMPLETE',work/'prediction.glb',flush=True)

if __name__ == '__main__':main()
