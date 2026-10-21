"""Exercise the decoder contract without importing CUDA extensions on this Mac."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch


def decode_function():
    source=Path(__file__).parents[1]/'uniforge3d/pipelines/texture.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode_shape_and_texture')
    namespace={'torch':torch,'SparseTensor':object}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['decode_shape_and_texture']


def test_hole_filling_can_wait_until_decoders_are_off_gpu():
    events=[]
    mesh=SimpleNamespace(fill_holes=lambda:events.append('holes'))
    def shape_decoder(latent,return_subs):
        events.append('shape');return [mesh],'shared-subdivisions'
    def texture_decoder(latent,guide_subs):
        assert guide_subs=='shared-subdivisions'
        events.append('texture');return torch.tensor([[0.2,0.6]])
    result,voxel=decode_function()(shape_decoder,texture_decoder,object(),object(),fill_holes=False)
    assert result is mesh and events==['shape','texture']
    torch.testing.assert_close(voxel,torch.tensor([0.6,0.8]))
    events.append('decoders-offloaded')
    result.fill_holes()
    assert events==['shape','texture','decoders-offloaded','holes']


def test_default_decoder_still_fills_holes():
    events=[]
    mesh=SimpleNamespace(fill_holes=lambda:events.append('holes'))
    result,voxel=decode_function()(lambda *a,**kw:([mesh],None),lambda *a,**kw:torch.tensor([[0.]]),None,None)
    assert result is mesh and events==['holes']
    torch.testing.assert_close(voxel,torch.tensor([.5]))
