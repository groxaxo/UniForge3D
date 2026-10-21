import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def reducer():
    path=Path(__file__).parents[1]/'uniforge3d/utils/mesh_budget.py'
    spec=importlib.util.spec_from_file_location('mesh_budget_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.cpu_reduce_if_needed


def tetra():
    return torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],dtype=torch.int32)


def test_small_mesh_does_not_require_a_cpu_dependency():
    v,f=tetra();a,b=reducer()(v,f,4)
    assert a is v and b is f


def test_oversized_mesh_returns_finite_indices_in_original_device_and_dtype(monkeypatch):
    v,f=tetra();calls=[]
    def simplify(points,triangles,target_count,agg):
        calls.append((target_count,agg));return points,triangles[:target_count]
    monkeypatch.setitem(sys.modules,'fast_simplification',SimpleNamespace(simplify=simplify))
    a,b=reducer()(v,f,2)
    assert calls==[(2,7.0)] and len(b)==2
    assert a.dtype==v.dtype and b.dtype==f.dtype and a.device==v.device
    assert torch.equal(a,v) and torch.equal(b,f[:2])


@pytest.mark.parametrize('invalid',['nan','indices','budget'])
def test_invalid_cpu_reduction_never_reaches_cuda(monkeypatch,invalid):
    v,f=tetra()
    def simplify(points,triangles,**kwargs):
        points=points.copy();triangles=triangles[:2].copy()
        if invalid=='nan':points[0,0]=np.nan
        if invalid=='indices':triangles[0,0]=1000
        if invalid=='budget':triangles=np.tile(triangles,(3,1))
        return points,triangles
    monkeypatch.setitem(sys.modules,'fast_simplification',SimpleNamespace(simplify=simplify))
    with pytest.raises(ValueError):reducer()(v,f,2)
