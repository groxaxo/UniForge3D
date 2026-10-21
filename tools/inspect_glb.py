"""Validate GLB geometry and render four real mesh views using CUDA rasterization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import trimesh


def inspect(path, render=True):
    scene=trimesh.load(path,force='scene',process=False)
    meshes=list(scene.dump())
    assert meshes and all(isinstance(m,trimesh.Trimesh) for m in meshes)
    vertices=sum(len(m.vertices) for m in meshes)
    faces=sum(len(m.faces) for m in meshes)
    bounds=np.stack([np.min([m.bounds[0] for m in meshes],axis=0),np.max([m.bounds[1] for m in meshes],axis=0)])
    assert vertices>100 and faces>100 and path.stat().st_size>10000
    assert np.isfinite(bounds).all() and (bounds[1]-bounds[0]).max()>0
    materials=[]
    for m in meshes:
        assert np.isfinite(m.vertices).all()
        assert m.faces.min()>=0 and m.faces.max()<len(m.vertices)
        mat=getattr(m.visual,'material',None)
        tex=getattr(mat,'baseColorTexture',None)
        materials.append({'name':getattr(mat,'name',None),'texture_size':list(tex.size) if tex else None})
    result={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'geometry_count':len(meshes),'vertices':vertices,'faces':faces,'bounds':bounds.tolist(),
        'extents':(bounds[1]-bounds[0]).tolist(),'finite_vertices':True,'valid_face_indices':True,
        'materials':materials}
    if render:
        render_meshes(meshes,bounds,path.with_suffix('.preview.png'))
        result['preview']=str(path.with_suffix('.preview.png'))
    path.with_suffix('.stats.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def render_meshes(meshes,bounds,path):
    import torch
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda')
    center=bounds.mean(axis=0)
    extent=(bounds[1]-bounds[0]).max()
    size=640
    canvas=Image.new('RGB',(size*4,size+32),'white')
    fov=np.deg2rad(40)
    near,far=0.01,10.
    proj=np.array([[1/np.tan(fov/2),0,0,0],[0,1/np.tan(fov/2),0,0],
        [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],dtype=np.float32)
    for index,deg in enumerate([0,90,180,270]):
        a=np.deg2rad(deg)
        eye=np.array([np.sin(a)*2.1,0.25,np.cos(a)*2.1])*extent+center
        forward=(center-eye);forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        view=np.eye(4);view[:3,:3]=np.stack([right,up,-forward]);view[:3,3]=-view[:3,:3]@eye
        vp=torch.tensor(proj@view,dtype=torch.float32,device='cuda')
        rgb=torch.ones((size,size,3),device='cuda')
        depth=torch.full((size,size),float('inf'),device='cuda')
        for mesh in meshes:
            v=torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32,device='cuda')
            f=torch.tensor(np.asarray(mesh.faces),dtype=torch.int32,device='cuda')
            vh=torch.cat([v,torch.ones((len(v),1),device='cuda')],dim=1)
            clip=(vh@vp.T)[None].contiguous()
            rast,_=dr.rasterize(context,clip,f,resolution=[size,size])
            mask=(rast[0,:,:,3]>0)&(rast[0,:,:,2]<depth)
            normal=torch.tensor(np.array(mesh.vertex_normals),dtype=torch.float32,device='cuda')
            normal,_=dr.interpolate(normal[None].contiguous(),rast,f)
            light=torch.tensor([0.3,0.7,0.65],device='cuda');light/=light.norm()
            shade=(normal[0]*light).sum(-1).abs().clamp(0,1)*0.65+0.35
            color=torch.full((size,size,3),0.72,device='cuda')
            uv=getattr(mesh.visual,'uv',None)
            mat=getattr(mesh.visual,'material',None)
            tex=getattr(mat,'baseColorTexture',None)
            if uv is not None and tex is not None:
                image=np.flipud(np.array(tex.convert('RGB'),dtype=np.float32)/255.).copy()
                texture=torch.tensor(image,device='cuda')[None].contiguous()
                uv=torch.tensor(np.array(uv),dtype=torch.float32,device='cuda')[None]
                texcoord,_=dr.interpolate(uv.contiguous(),rast,f)
                color=dr.texture(texture,texcoord,filter_mode='linear',boundary_mode='clamp')[0]
            rgb[mask]=(color*shade[:,:,None])[mask]
            depth[mask]=rast[0,:,:,2][mask]
        pixels=(rgb.flip(0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels),(index*size,32))
        ImageDraw.Draw(canvas).text((index*size+16,8),f'{deg} degrees',fill='black')
    canvas.save(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('glb',type=Path);p.add_argument('--no-render',action='store_true')
    a=p.parse_args();inspect(a.glb,not a.no_render)
