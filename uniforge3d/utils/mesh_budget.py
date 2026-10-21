"""Bound oversized intermediate meshes before native CUDA topology allocation."""
import json
import numpy as np
import torch


def cpu_reduce_if_needed(vertices, faces, face_budget):
    if face_budget is None or len(faces) <= face_budget:
        return vertices, faces
    if face_budget < 1:
        raise ValueError('Mesh face budget must be positive')
    import fast_simplification
    points, triangles = fast_simplification.simplify(
        vertices.detach().cpu().double().numpy(),
        faces.detach().cpu().int().numpy(),
        target_count=int(face_budget), agg=7.0,
    )
    if (points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3
            or not len(points) or not len(triangles) or len(triangles) > face_budget
            or not np.isfinite(points).all() or triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError('CPU mesh reduction did not produce valid geometry within its budget')
    print(json.dumps({'event': 'cpu_mesh_budget', 'input_faces': len(faces),
                      'output_faces': len(triangles), 'face_budget': face_budget}), flush=True)
    return (torch.as_tensor(points, dtype=vertices.dtype, device=vertices.device),
            torch.as_tensor(triangles, dtype=faces.dtype, device=faces.device))
